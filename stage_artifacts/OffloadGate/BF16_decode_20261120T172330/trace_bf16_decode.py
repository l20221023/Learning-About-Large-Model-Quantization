#!/usr/bin/env python3
"""Capture native official-BF16 router choices during prefill and greedy decode."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import torch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("this gate is pinned to physical CUDA device 0")

project_src = (Path(os.environ["P2_SRC_PATH"]) if "P2_SRC_PATH" in os.environ
               else Path(__file__).resolve().parents[4] / "src")
sys.path.insert(0, str(project_src))

from p2.weights.audit import write_json
from p2.weights.safetensors_io import now
from p2.trace.router import checked_routes

REPO = "deepseek-ai/DeepSeek-V2-Lite-Chat"
REV = "85864749cd611b4353ce1decdb286193298f64c7"
EVIDENCE = "MEASURED_OFFICIAL_BF16_V1_UVA20"
SEED = 20261117
LAYERS = tuple(range(1, 27))
EXPERTS = 64
TOP_K = 6


def digest_logits(items):
    h = hashlib.sha256()
    for item in items:
        x = item.float().contiguous().numpy()
        h.update(str(tuple(x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def install_head_probe(model):
    original = model.compute_logits
    model._offload_logits = []
    model._offload_logits_enabled = False

    def compute_logits(*args, **kwargs):
        value = original(*args, **kwargs)
        if value is not None and model._offload_logits_enabled:
            model._offload_logits.append(value.detach().to("cpu").contiguous().clone())
        return value

    model.compute_logits = compute_logits
    return type(model).__name__


def inspect_layers(model):
    layers = model.model.layers
    if len(layers) != 27 or hasattr(layers[0].mlp, "experts"):
        raise ValueError("unexpected DeepSeek-V2-Lite decoder topology")
    inventory = []
    for layer in LAYERS:
        block = layers[layer]
        runner = block.mlp.experts
        router = getattr(runner, "router", None)
        quant = runner.routed_experts.quant_method
        if router is None:
            raise ValueError(f"layer {layer}: native router object missing")
        inventory.append({"layer_id": layer, "router_class": type(router).__name__,
                          "quant_method": type(quant).__name__,
                          "monolithic": bool(quant.is_monolithic),
                          "backend": str(getattr(quant, "unquantized_backend", None)),
                          "gate_out_dtype": str(getattr(block.mlp.gate, "out_dtype", None))})
    return inventory


def install_router_probe(model):
    model._offload_routes = {layer: [] for layer in LAYERS}
    model._offload_route_enabled = False
    model._offload_router_originals = {}
    model._offload_router_modules = {}
    for layer in LAYERS:
        router = model.model.layers[layer].mlp.experts.router
        original = router.select_experts
        model._offload_router_originals[layer] = original
        model._offload_router_modules[layer] = router

        def wrap(layer_id, fn):
            def select(*args, **kwargs):
                weights, ids = fn(*args, **kwargs)
                if model._offload_route_enabled:
                    logits = kwargs.get("router_logits")
                    if logits is None:
                        raise ValueError(f"layer {layer_id}: router_logits missing from keyword arguments")
                    if logits.ndim != 2 or logits.shape[1] != EXPERTS:
                        raise ValueError(f"layer {layer_id}: expected [tokens,64] router logits, got {tuple(logits.shape)}")
                    if ids.shape != (logits.shape[0], TOP_K) or weights.shape != ids.shape:
                        raise ValueError(f"layer {layer_id}: native top-k shape mismatch")
                    logit_dtype = logits.dtype
                    if str(logit_dtype) not in ("torch.bfloat16", "torch.float16", "torch.float32"):
                        raise ValueError(f"layer {layer_id}: unsupported router dtype {logit_dtype}")
                    cpu_logits = logits.detach().cpu().contiguous()
                    cpu_ids = ids.detach().cpu().long().contiguous()
                    cpu_weights = weights.detach().cpu().float().contiguous()
                    # Do not replace native choices. Recompute candidate score paths
                    # only to validate the IDs/weights returned by vLLM unchanged.
                    candidates = [torch.float32, logit_dtype]
                    validation_dtype = None
                    last_error = None
                    for candidate in dict.fromkeys(candidates):
                        try:
                            probabilities = checked_routes(
                                cpu_logits, cpu_ids, cpu_weights, softmax_dtype=candidate)
                            validation_dtype = candidate
                            break
                        except (ValueError, RuntimeError) as exc:
                            last_error = exc
                    if validation_dtype is None:
                        raise ValueError(f"layer {layer_id}: native Top-6 did not match FP32 or router-dtype softmax: {last_error}")
                    selected_probabilities = probabilities.gather(1, cpu_ids)
                    model._offload_routes[layer_id].append({
                        "dtype": str(logit_dtype),
                        "validation_softmax_dtype": str(validation_dtype),
                        "weights_dtype": str(weights.dtype),
                        "logits": cpu_logits.float().tolist(),
                        "ids": cpu_ids.tolist(),
                        "weights": cpu_weights.tolist(),
                        "probabilities": selected_probabilities.tolist(),
                    })
                return weights, ids
            return select

        router.select_experts = wrap(layer, original)
    return True


def remove_router_probe(model):
    model._offload_route_enabled = False
    for layer in LAYERS:
        model._offload_router_modules[layer].select_experts = model._offload_router_originals[layer]


def run_once(llm, model, prompt_ids, sampling, *, trace, torch):
    if trace:
        model._offload_routes = {layer: [] for layer in LAYERS}
        model._offload_route_enabled = True
    torch.cuda.synchronize()
    started = perf_counter()
    try:
        result, = llm.generate([{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        model._offload_route_enabled = False
    output, = result.outputs
    elapsed = perf_counter() - started
    return result, list(output.token_ids), output.text, output.finish_reason, [], model._offload_routes if trace else None, elapsed


def validate_parity(sample_id, baseline, traced):
    b, b_ids, b_text, b_finish, _, _, baseline_seconds = baseline
    t, t_ids, t_text, t_finish, _, _, instrumented_seconds = traced
    if b_ids != t_ids or b_text != t_text or b_finish != t_finish:
        raise ValueError(f"{sample_id}: baseline and instrumented greedy outputs differ")
    if not b_ids:
        raise ValueError(f"{sample_id}: generation returned no tokens")
    return {"token_ids_equal": True, "text_equal": True, "finish_reason_equal": True,
            "baseline_token_ids": b_ids, "instrumented_token_ids": t_ids,
            "baseline_seconds": baseline_seconds,
            "instrumented_seconds": instrumented_seconds,
            "added_seconds": instrumented_seconds - baseline_seconds,
            "relative_overhead": instrumented_seconds / baseline_seconds - 1.0,
            "baseline_text_sha256": hashlib.sha256(b_text.encode()).hexdigest(),
            "instrumented_text_sha256": hashlib.sha256(t_text.encode()).hexdigest(),
            "generated_token_ids": b_ids, "generated_text": b_text,
            "finish_reason": b_finish}


def rows_for_request(sample_id, group, prompt_ids, result, routes, request_kind):
    _, generated_ids, _, _, _, _, _ = result
    expected_calls = len(generated_ids)
    if set(routes) != set(LAYERS):
        raise ValueError(f"{sample_id}: routed layer set differs")
    rows = []
    for layer in LAYERS:
        calls = routes[layer]
        if len(calls) != expected_calls:
            raise ValueError(f"{sample_id}/layer{layer}: expected {expected_calls} router calls, got {len(calls)}")
        prefill = calls[0]
        if len(prefill["ids"]) != len(prompt_ids):
            raise ValueError(f"{sample_id}/layer{layer}: prefill rows do not match templated prompt tokens")
        for position, token_id in enumerate(prompt_ids):
            rows.append({"evidence": EVIDENCE, "sample_id": sample_id, "workload_group": group,
                         "request_kind": request_kind, "phase": "prefill", "step": None,
                         "layer_id": layer, "call_index": 0, "sequence_index": position,
                         "token_id": token_id, "input_token_id": token_id,
                         "predicts_output_index": 0,
                         "router_logits_dtype": prefill["dtype"],
                         "route_validation_softmax_dtype": prefill["validation_softmax_dtype"],
                         "topk_weights_dtype": prefill["weights_dtype"],
                         "router_logits": prefill["logits"][position],
                         "topk_expert_ids": prefill["ids"][position],
                         "topk_weights": prefill["weights"][position],
                         "topk_probabilities": prefill["probabilities"][position]})
        for call_index, call in enumerate(calls[1:], start=1):
            if len(call["ids"]) != 1:
                raise ValueError(f"{sample_id}/layer{layer}: decode call {call_index} was not one token")
            step = call_index - 1
            token_id = generated_ids[step]
            rows.append({"evidence": EVIDENCE, "sample_id": sample_id, "workload_group": group,
                         "request_kind": request_kind, "phase": "decode", "step": step,
                         "layer_id": layer, "call_index": call_index, "sequence_index": None,
                         "token_id": token_id, "input_token_id": token_id,
                         "predicts_output_index": step + 1,
                         "router_logits_dtype": call["dtype"],
                         "route_validation_softmax_dtype": call["validation_softmax_dtype"],
                         "topk_weights_dtype": call["weights_dtype"],
                         "router_logits": call["logits"][0],
                         "topk_expert_ids": call["ids"][0],
                         "topk_weights": call["weights"][0],
                         "topk_probabilities": call["probabilities"][0]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=32)
    args = parser.parse_args()
    if not 4 <= args.decode_tokens <= 64:
        raise ValueError("decode token cap must be between 4 and 64")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    workload_bytes = args.workload.read_bytes()
    workload = json.loads(workload_bytes)
    snapshot = args.snapshot.resolve(strict=True)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("VLLM_USE_V2_MODEL_RUNNER must be explicitly 0")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("VLLM_ENABLE_V1_MULTIPROCESSING must be explicitly 0")

    options = {"dtype": "bfloat16", "max_model_len": 256, "max_num_batched_tokens": 256,
               "max_num_seqs": 1, "enforce_eager": True, "enable_prefix_caching": False,
               "trust_remote_code": False, "tensor_parallel_size": 1,
               "kv_cache_memory_bytes": 512 * 1024**2, "seed": SEED,
               "offload_backend": "uva", "cpu_offload_gb": 20}
    manifest = {"started_at": now(), "model": REPO, "revision": REV,
                "evidence": EVIDENCE, "phase_scope": ["prefill", "continuous_decode"],
                "workload_id": workload["workload_id"],
                "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
                "decode_token_cap": args.decode_tokens,
                "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": args.decode_tokens,
                               "seed": SEED, "ignore_eos": False, "stop": None},
                "engine_options": options, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "physical_gpu_index": 0, "tensor_parallel_size": 1,
                "status": "FAIL", "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
                "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}
    status = 1
    try:
        import torch
        from vllm import LLM, SamplingParams, __version__

        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
            raise ValueError(f"expected one visible RTX 3090, got count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")
        if __version__ != "0.26.0":
            raise ValueError(f"expected vLLM 0.26.0, got {__version__}")
        config = json.loads((snapshot / "config.json").read_bytes())
        expected_config = {"n_routed_experts": 64, "num_experts_per_tok": 6,
                           "topk_method": "greedy", "scoring_func": "softmax",
                           "norm_topk_prob": False, "routed_scaling_factor": 1.0}
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise ValueError(f"checkpoint gate config mismatch: {key}={config.get(key)!r}, expected {expected!r}")
        manifest.update(vllm_version=__version__, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        gpu_name=torch.cuda.get_device_name(0), quantization_config=config.get("quantization_config"),
                        gate_config={key: config.get(key) for key in expected_config})
        t0 = perf_counter()
        llm = LLM(model=str(snapshot), **options)
        manifest["model_init_seconds"] = perf_counter() - t0
        tokenizer = llm.get_tokenizer()
        model, = llm.apply_model(lambda model: model)
        inventory = inspect_layers(model)
        manifest["router_layers"] = inventory
        if any(row["monolithic"] for row in inventory):
            raise RuntimeError("BF16 MoE uses a monolithic path that bypasses router.select_experts; refusing to recompute or alter routing")
        if any(row["quant_method"] != "UnquantizedFusedMoEMethod" for row in inventory):
            raise RuntimeError("expected official BF16 UnquantizedFusedMoEMethod on every routed layer")
        from vllm.model_executor.offloader import base as offloader_base
        offloader = offloader_base.get_offloader()
        offloader_name = type(offloader).__name__
        offload_bytes = int(getattr(offloader, "cpu_offload_bytes", 0))
        manifest["offloader"] = {"class": offloader_name, "cpu_offload_bytes": offload_bytes,
                                  "uva_offloading": getattr(offloader, "uva_offloading", None)}
        if offloader_name != "UVAOffloader" or offload_bytes <= 0:
            raise RuntimeError(f"UVA 20 GiB path not active after model load: {offloader_name}, bytes={offload_bytes}")
        model_runner = getattr(getattr(getattr(llm, "llm_engine", None), "model_executor", None),
                               "driver_worker", None)
        model_runner = getattr(model_runner, "model_runner", None)
        runner_name = type(model_runner).__name__ if model_runner is not None else None
        runner_module = type(model_runner).__module__ if model_runner is not None else None
        manifest["model_runner"] = {"class": runner_name, "module": runner_module}
        if model_runner is None:
            raise RuntimeError("cannot inspect the active model runner to verify V1 selection")
        if "v2" in runner_module.lower() or "V2" in runner_name:
            raise RuntimeError(f"V2 model runner active despite V1 request: {runner_module}.{runner_name}")
        install_router_probe(model)
        sampling_smoke = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4, seed=SEED)
        sampling_full = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.decode_tokens, seed=SEED)

        ordered_requests = []
        for group, prompts in workload["groups"].items():
            for index, prompt in enumerate(prompts, 1):
                sample_id = f"{group}-{index:02d}"
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=False)
                if not isinstance(prompt_ids, list) or not prompt_ids:
                    raise ValueError(f"{sample_id}: tokenizer returned no prompt IDs")
                ordered_requests.append((sample_id, group, prompt, prompt_ids))

        # A real short baseline/trace comparison gates the longer set.
        sample_id, group, prompt, prompt_ids = ordered_requests[0]
        smoke_base = run_once(llm, model, prompt_ids, sampling_smoke, trace=False, torch=torch)
        try:
            smoke_trace = run_once(llm, model, prompt_ids, sampling_smoke, trace=True, torch=torch)
        finally:
            remove_router_probe(model)
        smoke_parity = validate_parity("smoke-" + sample_id, smoke_base, smoke_trace)
        smoke_rows = rows_for_request("smoke-" + sample_id, group, prompt_ids, smoke_trace,
                                      smoke_trace[5], "smoke")
        with (out / "smoke_router_calls.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in sorted(smoke_rows, key=lambda row: (0 if row["phase"] == "prefill" else 1,
                    row["step"] if row["step"] is not None else row["sequence_index"], row["layer_id"])):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        write_json(out / "smoke.json", {"sample_id": sample_id, "prompt": prompt,
                   "prompt_token_ids": prompt_ids, "parity": smoke_parity,
                   "router_calls": len(smoke_rows), "decode_steps": max(0, len(smoke_parity["generated_token_ids"]) - 1),
                   "baseline_seconds": smoke_base[6], "instrumented_seconds": smoke_trace[6]})
        print(json.dumps({"event": "SMOKE_PASS", "sample_id": sample_id,
                          "output_tokens": len(smoke_parity["generated_token_ids"]),
                          "baseline_seconds": smoke_base[6],
                          "instrumented_seconds": smoke_trace[6]}, sort_keys=True), flush=True)

        all_rows = []
        request_summaries = []
        timing_rows = []
        t0 = perf_counter()
        print(json.dumps({"event": "FORMAL_START", "request_count": len(ordered_requests),
                          "max_tokens": args.decode_tokens}, sort_keys=True), flush=True)
        for sample_id, group, prompt, prompt_ids in ordered_requests:
            baseline = run_once(llm, model, prompt_ids, sampling_full, trace=False, torch=torch)
            install_router_probe(model)
            try:
                traced = run_once(llm, model, prompt_ids, sampling_full, trace=True, torch=torch)
            finally:
                remove_router_probe(model)
            parity = validate_parity(sample_id, baseline, traced)
            timing_rows.append({"sample_id": sample_id, "baseline_seconds": baseline[6],
                                "instrumented_seconds": traced[6],
                                "added_seconds": traced[6] - baseline[6],
                                "relative_overhead": traced[6] / baseline[6] - 1.0})
            rows = rows_for_request(sample_id, group, prompt_ids, traced, traced[5], "formal")
            all_rows.extend(rows)
            with (out / "router_calls.inprogress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
            decode_count = sum(row["phase"] == "decode" for row in rows)
            request_summaries.append({"sample_id": sample_id, "workload_group": group,
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_ids": prompt_ids, "prompt_tokens": len(prompt_ids),
                "generation": manifest["generation"], "parity": parity,
                "baseline_seconds": baseline[6], "instrumented_seconds": traced[6],
                "added_seconds": traced[6] - baseline[6],
                "relative_overhead": traced[6] / baseline[6] - 1.0,
                "prefill_router_rows": len(prompt_ids) * len(LAYERS),
                "decode_steps": max(0, len(parity["generated_token_ids"]) - 1),
                "decode_router_rows": decode_count,
                "router_calls_per_layer": len(parity["generated_token_ids"]),
                "status": "PASS"})
            print(json.dumps({"event": "FORMAL_REQUEST_PASS", "sample_id": sample_id,
                              "output_tokens": len(parity["generated_token_ids"]),
                              "baseline_seconds": baseline[6],
                              "instrumented_seconds": traced[6]}, sort_keys=True), flush=True)
        manifest["formal_seconds"] = perf_counter() - t0
        all_rows.sort(key=lambda row: (row["sample_id"], 0 if row["phase"] == "prefill" else 1,
                                       row["step"] if row["step"] is not None else row["sequence_index"], row["layer_id"]))
        os.replace(out / "router_calls.inprogress.jsonl", out / "router_calls_capture_order.jsonl")
        decode_rows = sorted((row for row in all_rows if row["phase"] == "decode"),
                             key=lambda row: (row["sample_id"], row["step"], row["layer_id"]))
        prefill_rows = [row for row in all_rows if row["phase"] == "prefill"]
        with (out / "router_calls.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        with (out / "decode_trace.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in decode_rows:
                # This field set is accepted by stage_artifacts/OffloadGate/simulate_cache.py.
                simulation_row = {"sample_id": row["sample_id"], "step": row["step"],
                                  "layer_id": row["layer_id"], "topk_expert_ids": row["topk_expert_ids"],
                                  "token_id": row["token_id"], "predicts_output_index": row["predicts_output_index"],
                                  "router_logits_dtype": row["router_logits_dtype"],
                                  "topk_weights": row["topk_weights"]}
                stream.write(json.dumps(simulation_row, separators=(",", ":"), allow_nan=False) + "\n")
        with (out / "prefill_trace.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in prefill_rows:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        with (out / "requests.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in request_summaries:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        write_json(out / "timings.json", {
            "request_timings": timing_rows,
            "baseline_total_seconds": sum(r["baseline_seconds"] for r in timing_rows),
            "instrumented_total_seconds": sum(r["instrumented_seconds"] for r in timing_rows),
            "added_total_seconds": sum(r["instrumented_seconds"] - r["baseline_seconds"] for r in timing_rows),
            "relative_overhead": sum(r["instrumented_seconds"] for r in timing_rows)
                                 / sum(r["baseline_seconds"] for r in timing_rows) - 1.0,
            "scope": "per-request generate wall time with CUDA synchronization; baseline has no route or logits hook",
        })
        manifest.update(status="PASS", exit_code=0, request_count=len(request_summaries),
            router_layers=list(LAYERS), expert_count=EXPERTS, top_k=TOP_K,
            prefill_rows=len(prefill_rows), decode_rows=len(decode_rows),
            decode_steps_by_request={row["sample_id"]: row["decode_steps"] for row in request_summaries},
            output_files={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                          for name in ("router_calls.jsonl", "decode_trace.jsonl", "prefill_trace.jsonl", "requests.jsonl",
                                       "router_calls_capture_order.jsonl", "smoke_router_calls.jsonl", "timings.json")},
            invariants={"baseline_equals_instrumented_token_ids_text_and_finish_reason": True,
                        "each_layer_prefill_rows_equal_templated_prompt_ids": True,
                        "each_decode_router_call_has_one_input_token": True,
                        "all_native_topk_rows_validated": True},
            cache_sim_input_schema="sample_id,step,layer_id,topk_expert_ids; one row is one layer-step atomic Top-6 set",
            decode_step_semantics="step 0 consumes generated output token 0 and predicts output token 1; prefill predicts output token 0")
        status = 0
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest.update(exit_code=status, finished_at=now())
        write_json(out / "manifest.json", manifest)
    print(json.dumps({key: manifest.get(key) for key in
                      ("status", "exit_code", "request_count", "prefill_rows", "decode_rows", "error")}, sort_keys=True))


if __name__ == "__main__":
    main()


