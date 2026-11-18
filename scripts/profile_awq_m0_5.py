"""Measure same-token routed expert outputs on one AWQ MoE layer."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

_here = Path(__file__).resolve().parent
_source_root = _here / "src" if (_here / "src").is_dir() else _here.parent / "src"
sys.path.insert(0, str(_source_root))

from p2.experts.similarity import (fixed_weight_replacement_relative_l2,
                                   output_similarity, reconstruct_routed_output)
from p2.weights.audit import write_json
from p2.weights.safetensors_io import now
try:
    from .trace_awq_3090 import EVIDENCE, REPO, REV, SEED
except ImportError:
    from trace_awq_3090 import EVIDENCE, REPO, REV, SEED


def inventory(model, layer_id):
    layers = model.model.layers
    if len(layers) != 27 or not 1 <= layer_id <= 26:
        raise ValueError("unexpected DeepSeek-V2-Lite decoder topology or layer")
    runner = layers[layer_id].mlp.experts
    quant = runner.routed_experts.quant_method
    if runner.moe_config.tp_size != 1 or runner.moe_config.dp_size != 1:
        raise ValueError("routed output capture requires single-device execution")
    if quant.is_monolithic:
        raise ValueError("cannot shadow-route experts through a monolithic MoE kernel")
    kernel = getattr(quant, "moe_kernel", None)
    implementation = getattr(kernel, "impl", None)
    fused = getattr(implementation, "fused_experts", None)
    return runner, {"layer_id": layer_id, "runner": type(runner).__name__,
                    "router": type(runner.router).__name__,
                    "quant_method": type(quant).__name__,
                    "kernel": type(kernel).__name__ if kernel is not None else None,
                    "fused_experts": type(fused).__name__ if fused is not None else None}


def single_worker(results):
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one vLLM model worker, got {type(results).__name__}")
    return results[0]


def moe_hidden_from_call(args, kwargs):
    if args:
        hidden = args[0]
    else:
        hidden = kwargs.get("hidden_states")
    if hidden is None or not hasattr(hidden, "ndim") or hidden.ndim != 2:
        raise ValueError("FusedMoE call did not supply 2-D hidden_states")
    return hidden


def run_shadow(runner, hidden, router_logits):
    if hidden.ndim != 1 or router_logits.ndim != 1:
        raise ValueError("shadow inputs must be one-token vectors")
    if hidden.device != router_logits.device:
        raise ValueError("shadow hidden and router logits must share device")
    return runner(hidden_states=hidden.unsqueeze(0),
                  router_logits=router_logits.unsqueeze(0))


def capture_layer(model, layer_id):
    from dataclasses import replace
    from vllm.forward_context import get_forward_context

    model._p2_m05 = {"hidden": None, "native_output": None, "native_route": None}
    runner = model.model.layers[layer_id].mlp.experts

    def pre_hook(_module, inputs, kwargs):
        if model._p2_m05["hidden"] is not None:
            raise ValueError("target FusedMoE called more than once for request")
        hidden = moe_hidden_from_call(inputs, kwargs)
        if hidden.ndim != 2 or hidden.shape[0] < 1:
            raise ValueError(f"unexpected FusedMoE hidden layout {tuple(hidden.shape)}")
        model._p2_m05["hidden"] = hidden.detach().clone()

        model._p2_m05["context"] = replace(get_forward_context(), all_moe_layers=None)

    def output_hook(_module, _inputs, output):
        if model._p2_m05["native_output"] is None:
            raise ValueError("routed output capture did not run")

    handle_pre = runner.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handle_out = runner.register_forward_hook(output_hook)
    router = runner.router
    original = router.select_experts

    def select(*args, **kwargs):
        weights, ids = original(*args, **kwargs)
        if model._p2_m05["native_route"] is not None:
            raise ValueError("target router called more than once for request")
        logits = kwargs.get("router_logits")
        if logits is None:
            raise ValueError("vLLM router API did not pass router_logits by keyword")
        model._p2_m05["native_route"] = (logits.detach().clone(), ids.detach().clone(),
                                         weights.detach().clone())
        return weights, ids

    router.select_experts = select
    return runner, router, original, handle_pre, handle_out


def intercept_routed(runner, destination):
    original = runner._maybe_apply_routed_scale_to_output

    def record(shared, routed):
        shared, routed = original(shared, routed)
        destination["native_output"] = routed.detach().clone()
        return shared, routed

    runner._maybe_apply_routed_scale_to_output = record
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    workload_bytes = args.workload.read_bytes()
    workload = json.loads(workload_bytes)
    layer_id = workload["layer_id"]
    candidates = workload["candidate_experts"]
    expert_ids = [int(item["expert_id"]) for item in candidates]
    if len(set(expert_ids)) != len(expert_ids) or any(not 0 <= i < 64 for i in expert_ids):
        raise ValueError("candidate experts must be distinct IDs in 0..63")
    expected_pairs = {tuple(sorted((int(a), int(b)))) for a, b in workload["pairs"]}
    if expected_pairs != {tuple(sorted((a, b))) for i, a in enumerate(expert_ids)
                          for b in expert_ids[i + 1:]}:
        raise ValueError("expected pairs of candidate experts")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    meta = {"started_at": now(), "model": REPO, "revision": REV, "evidence": EVIDENCE,
            "workload_id": workload["workload_id"],
            "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
            "seed": workload["seed"], "layer_id": layer_id,
            "candidate_experts": candidates, "status": "FAIL"}
    captured = []
    handles = None
    status = 1
    try:
        import torch
        from vllm.forward_context import override_forward_context
        from vllm import LLM, SamplingParams, __version__
        if __version__ != "0.26.0":
            raise ValueError(f"instrumentation pinned to vLLM 0.26.0, got {__version__}")
        options = dict(dtype="half", max_model_len=256, max_num_batched_tokens=256,
                       max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False,
                       trust_remote_code=False, tensor_parallel_size=1,
                       kv_cache_memory_bytes=512 * 1024**2, seed=SEED)
        meta.update(vllm_version=__version__, torch_version=torch.__version__,
                    cuda=torch.version.cuda, engine_options=options)
        llm = LLM(model=str(args.snapshot.resolve(strict=True)), **options)
        meta.update(gpu=torch.cuda.get_device_name(), runtime_model_class=
                    single_worker(llm.apply_model(lambda model: type(model).__name__)))
        tokenizer = llm.get_tokenizer()
        special = set(tokenizer.all_special_ids)
        runner, layer_info = single_worker(llm.apply_model(lambda model: inventory(model, layer_id)))
        meta["runtime_layer"] = layer_info
        sampling = SamplingParams(temperature=0.0, max_tokens=1, seed=SEED)
        for sample in workload["samples"]:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=False)
            valid_positions = [i for i, token in enumerate(prompt_ids) if token not in special]
            if not valid_positions:
                raise ValueError(f"no non-special tokens for {sample['sample_id']}")
            positions = sorted(set([valid_positions[0], valid_positions[-1]]))
            runtime_runner, router, original, h_pre, h_out = single_worker(llm.apply_model(
                lambda model: capture_layer(model, layer_id)))
            if runtime_runner is not runner:
                raise ValueError("model runner changed between inventory and request")
            handles = (router, original, h_pre, h_out)
            state = single_worker(llm.apply_model(lambda model: model._p2_m05))
            original_scale = intercept_routed(runtime_runner, state)
            try:
                result, = llm.generate([{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False)
            finally:
                runtime_runner._maybe_apply_routed_scale_to_output = original_scale
            h_pre.remove(); h_out.remove()
            router.select_experts = original
            handles = None
            h, native_output = state["hidden"], state["native_output"]
            route_logits, native_ids, native_weights = state["native_route"]
            if h.shape[0] != len(prompt_ids) or route_logits.shape[0] != len(prompt_ids):
                raise ValueError("hidden/router token rows do not match templated prompt")
            details = []
            for pos in positions:
                selected_h = h[pos].detach()
                selected_ids = native_ids[pos].detach().cpu().tolist()
                selected_weights = native_weights[pos].detach().float().cpu().tolist()
                routed_expert_outputs = {}
                per_candidate = {}
                if len(selected_ids) != 6 or len(set(selected_ids)) != 6:
                    raise ValueError("native Top-6 router IDs are incomplete or duplicated")
                for candidate in dict.fromkeys([*expert_ids, *selected_ids]):
                    fill = [e for e in range(64) if e != candidate][:5]
                    forced_ids = torch.tensor([[candidate] + fill], device=h.device,
                                              dtype=native_ids.dtype)
                    forced_weights = torch.zeros((1, 6), device=h.device,
                                                 dtype=native_weights.dtype)
                    forced_weights[0, 0] = 1

                    def force(*args, _ids=forced_ids, _weights=forced_weights, **kwargs):
                        # Keep router API output on the same device/dtype and in native shape.
                        return _weights, _ids

                    router.select_experts = force
                    try:
                        shadow = {}
                        original_scale = intercept_routed(runtime_runner, shadow)
                        try:
                            with torch.inference_mode(), override_forward_context(state["context"]):
                                run_shadow(runtime_runner, selected_h.clone(),
                                           route_logits[pos].detach())
                        finally:
                            runtime_runner._maybe_apply_routed_scale_to_output = original_scale
                        isolated = shadow["native_output"].squeeze(0)
                    finally:
                        router.select_experts = original
                    if not torch.isfinite(isolated).all():
                        raise ValueError(f"expert {candidate} returned non-finite output")
                    per_candidate[candidate] = isolated
                    if candidate in selected_ids:
                        routed_expert_outputs[candidate] = isolated
                router.select_experts = original
                reconstructed = reconstruct_routed_output(routed_expert_outputs,
                                                          selected_ids, selected_weights)
                observed = native_output[pos].detach().float()
                parity = torch.allclose(reconstructed.to(observed.dtype), observed,
                                        atol=2e-3, rtol=2e-3)
                if not parity:
                    error = float(torch.linalg.vector_norm(reconstructed - observed) /
                                  torch.linalg.vector_norm(observed).clamp_min(1e-12))
                    raise ValueError(f"shadow reconstruction failed at {sample['sample_id']}:{pos}; relative_l2={error}")
                token_rows = []
                baseline_expert = selected_ids[0]
                baseline_weight = selected_weights[0]
                for candidate in expert_ids:
                    metrics = output_similarity(per_candidate[expert_ids[0]],
                                                per_candidate[candidate])
                    native_slots = [i for i, expert in enumerate(selected_ids) if expert == candidate]
                    router_weight = selected_weights[native_slots[0]] if native_slots else 0.0
                    replacement = fixed_weight_replacement_relative_l2(
                        observed, per_candidate[baseline_expert], per_candidate[candidate], baseline_weight)
                    row = {"evidence": EVIDENCE, "layer_id": layer_id,
                           "sample_id": sample["sample_id"],
                           "workload_group": sample["workload_group"], "token_position": pos,
                           "token_id": int(prompt_ids[pos]), "expert_id": candidate,
                           "frequency_class": next(x["frequency_class"] for x in candidates
                                                    if x["expert_id"] == candidate),
                           "native_selected": bool(native_slots), "native_router_weight": router_weight,
                           "cosine_similarity": metrics["cosine_similarity"],
                           "relative_l2": metrics["relative_l2"],
                           "left_norm": metrics["left_norm"], "right_norm": metrics["right_norm"],
                           "norm_ratio_right_over_left": metrics["norm_ratio_right_over_left"],
                           "absolute_l2": metrics["absolute_l2"],
                           "fixed_weight_replacement_relative_l2": replacement,
                           "shadow_reconstruction_allclose": parity,
                           "shadow_reconstruction_relative_l2": float(
                               torch.linalg.vector_norm(reconstructed - observed) /
                               torch.linalg.vector_norm(observed).clamp_min(1e-12))}
                    token_rows.append(row)
                pair_rows = []
                for left_index, left_id in enumerate(expert_ids):
                    for right_id in expert_ids[left_index + 1:]:
                        metrics = output_similarity(per_candidate[left_id], per_candidate[right_id])
                        pair_rows.append({"expert_a": left_id, "expert_b": right_id,
                                          **metrics})
                details.append({"token_position": pos, "token_id": int(prompt_ids[pos]),
                                "native_expert_ids": selected_ids,
                                "native_router_weights": selected_weights,
                                "shadow_reconstruction_pass": parity,
                                "rows": token_rows, "pair_metrics": pair_rows})
            captured.extend(row for detail in details for row in detail["rows"])
            write_json(out / f"{sample['sample_id']}.json", {
                "sample_id": sample["sample_id"], "prompt_token_ids": prompt_ids,
                "generated_token_ids": list(result.outputs[0].token_ids),
                "layer_id": layer_id, "tokens": details})
        pair_captured = []
        for sample in workload["samples"]:
            sample_result = json.loads((out / f"{sample['sample_id']}.json").read_text(encoding="utf-8"))
            for token in sample_result["tokens"]:
                for pair in token["pair_metrics"]:
                    pair_captured.append({"evidence": EVIDENCE, "layer_id": layer_id,
                                          "sample_id": sample["sample_id"],
                                          "workload_group": sample["workload_group"],
                                          "token_position": token["token_position"], **pair})
        (out / "expert_similarity.json").write_text(json.dumps(captured, allow_nan=False,
                                                                  separators=(",", ":")) + "\n",
                                                     encoding="utf-8")
        (out / "expert_pair_similarity.json").write_text(json.dumps(pair_captured, allow_nan=False,
                                                                       separators=(",", ":")) + "\n",
                                                          encoding="utf-8")
        meta.update(status="PASS", sample_count=len(workload["samples"]),
                    token_observations=sum(len(r["tokens"]) for r in [json.loads(
                        (out / f"{s['sample_id']}.json").read_text()) for s in workload["samples"]]),
                    row_count=len(captured), excluded_special_ids=sorted(special),
                    artifacts=["expert_similarity.json", "expert_pair_similarity.json"])
        status = 0
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if handles:
            router, original, h_pre, h_out = handles
            router.select_experts = original
            h_pre.remove(); h_out.remove()
        meta.update(finished_at=now(), exit_code=status)
        write_json(out / "manifest.json", meta)


if __name__ == "__main__":
    main()
