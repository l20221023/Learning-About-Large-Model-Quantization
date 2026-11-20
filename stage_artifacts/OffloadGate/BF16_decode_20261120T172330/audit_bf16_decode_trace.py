#!/usr/bin/env python3
"""Independent structural audit for the official BF16 decode trace."""
import argparse
import hashlib
import json
import math
from pathlib import Path

LAYERS = set(range(1, 27))
EXPERTS = set(range(64))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(trace_dir, workload_path):
    manifest = json.loads((trace_dir / "manifest.json").read_bytes())
    workload_bytes = workload_path.read_bytes()
    workload = json.loads(workload_bytes)
    require(manifest["status"] == "PASS" and manifest["exit_code"] == 0, "trace did not pass")
    require(manifest["model"] == "deepseek-ai/DeepSeek-V2-Lite-Chat"
            and manifest["revision"] == "85864749cd611b4353ce1decdb286193298f64c7",
            "official BF16 model revision mismatch")
    require(manifest["engine_options"]["dtype"] == "bfloat16"
            and manifest["engine_options"]["offload_backend"] == "uva"
            and manifest["engine_options"]["cpu_offload_gb"] == 20,
            "run was not BF16 + UVA 20 GiB")
    require(manifest["vllm_use_v2_model_runner"] == "0"
            and manifest["vllm_enable_v1_multiprocessing"] == "0",
            "V1 runtime environment mismatch")
    require(manifest["offloader"]["class"] == "UVAOffloader"
            and manifest["offloader"]["cpu_offload_bytes"] > 0,
            "UVA offloader was not active")
    runner = manifest["model_runner"]
    require(runner["class"] and runner["module"] and ".v2." not in runner["module"].lower(),
            "active model runner does not identify V1")
    require(manifest["workload_sha256"] == hashlib.sha256(workload_bytes).hexdigest(), "workload hash mismatch")
    require(manifest["workload_id"] == workload["workload_id"], "workload ID mismatch")
    require(manifest["phase_scope"] == ["prefill", "continuous_decode"], "phase scope mismatch")
    require(manifest["cuda_visible_devices"] == "0" and manifest["physical_gpu_index"] == 0,
            "trace is not pinned to physical GPU 0")
    require(manifest["tensor_parallel_size"] == 1, "unexpected tensor parallelism")
    router_layers = manifest["router_layers"]
    if router_layers and all(isinstance(x, int) for x in router_layers):
        router_layer_ids = router_layers
    else:
        router_layer_ids = [x["layer_id"] for x in router_layers]
        require(all(not x["monolithic"] and x["quant_method"] == "UnquantizedFusedMoEMethod"
                    for x in router_layers), "native router hook not available on all layers")
    require(router_layer_ids == sorted(LAYERS), "router layer inventory mismatch")
    requests = read_jsonl(trace_dir / "requests.jsonl")
    timings = json.loads((trace_dir / "timings.json").read_bytes())
    all_rows = read_jsonl(trace_dir / "router_calls.jsonl")
    decode_rows = read_jsonl(trace_dir / "decode_trace.jsonl")
    prefill_rows = read_jsonl(trace_dir / "prefill_trace.jsonl")
    expected_ids = {f"{group}-{idx:02d}": (group, prompt)
                    for group, prompts in workload["groups"].items()
                    for idx, prompt in enumerate(prompts, 1)}
    require(len(requests) == len(expected_ids) == manifest["request_count"], "request count mismatch")
    require(len(timings["request_timings"]) == len(requests), "timing count mismatch")
    request_by_id = {row["sample_id"]: row for row in requests}
    require(set(request_by_id) == set(expected_ids), "request IDs differ from workload")
    require(manifest["generation"]["temperature"] == 0.0 and manifest["generation"]["max_tokens"] == 32,
            "unexpected greedy generation settings")
    require(manifest["generation"]["seed"] == 20261117 and manifest["generation"]["ignore_eos"] is False,
            "unexpected generation seed or EOS behavior")

    seen_all, seen_decode, seen_prefill = set(), set(), set()
    expected_decode = set()
    expected_prefill_count = 0
    for sample_id, (group, prompt) in expected_ids.items():
        req = request_by_id[sample_id]
        require(req["workload_group"] == group and req["prompt"] == prompt, f"prompt mismatch: {sample_id}")
        require(req["status"] == "PASS", f"request failed: {sample_id}")
        parity = req["parity"]
        require(parity["token_ids_equal"] and parity["text_equal"] and parity["finish_reason_equal"],
                f"baseline parity failed: {sample_id}")
        require(parity["baseline_token_ids"] == parity["instrumented_token_ids"]
                == parity["generated_token_ids"], f"token ID parity failed: {sample_id}")
        require(parity["baseline_text_sha256"] == parity["instrumented_text_sha256"],
                f"text hash parity failed: {sample_id}")
        outputs = parity["generated_token_ids"]
        require(req["baseline_seconds"] > 0 and req["instrumented_seconds"] > 0,
                f"missing generation timings: {sample_id}")
        require(math.isfinite(req["parity"]["relative_overhead"]),
                f"missing instrumentation overhead: {sample_id}")
        require(len(outputs) >= 2, f"request has no continuous decode step: {sample_id}")
        require(req["router_calls_per_layer"] == len(outputs), f"router call count mismatch: {sample_id}")
        require(req["decode_steps"] == len(outputs) - 1, f"decode step count mismatch: {sample_id}")
        require(req["decode_router_rows"] == (len(outputs) - 1) * 26,
                f"decode row count mismatch: {sample_id}")
        require(req["prefill_router_rows"] == req["prompt_tokens"] * 26,
                f"prefill row count mismatch: {sample_id}")
        expected_prefill_count += req["prompt_tokens"] * 26
        for step in range(len(outputs) - 1):
            for layer in LAYERS:
                expected_decode.add((sample_id, step, layer, outputs[step], step + 1))

    require({row["layer_id"] for row in all_rows} == LAYERS,
            "native router calls do not cover all 26 layers")
    for row in all_rows:
        sample = row["sample_id"]
        require(sample in request_by_id, f"unexpected sample {sample}")
        require(row["layer_id"] in LAYERS, f"invalid layer {row['layer_id']}")
        ids, weights, probs, logits = (row["topk_expert_ids"], row["topk_weights"],
                                       row["topk_probabilities"], row["router_logits"])
        require(len(ids) == len(set(ids)) == len(weights) == len(probs) == 6,
                f"invalid Top-6 shape or duplicate IDs at {sample}")
        require(set(ids) <= EXPERTS, f"expert ID outside 0..63 at {sample}")
        require(len(logits) == 64 and all(math.isfinite(float(x)) for x in logits),
                f"invalid router logits at {sample}")
        require(all(math.isfinite(float(x)) and 0 <= x <= 1 for x in weights + probs),
                f"invalid selected weights/probabilities at {sample}")
        require(row["router_logits_dtype"] in ("torch.bfloat16", "torch.float16", "torch.float32"),
                f"unexpected router logits dtype at {sample}")
        require(row["route_validation_softmax_dtype"] in ("torch.bfloat16", "torch.float16", "torch.float32"),
                f"unexpected route validation dtype at {sample}")
        require(row["topk_weights_dtype"] == "torch.float32",
                f"unexpected native top-k weights dtype at {sample}")
        if row["phase"] == "prefill":
            key = (sample, row["sequence_index"], row["layer_id"])
            all_key = ("prefill", *key)
            require(row["step"] is None and row["token_id"] == row["input_token_id"],
                    f"bad prefill token alignment: {sample}")
            require(key not in seen_prefill, f"duplicate prefill token/layer: {key}")
            seen_prefill.add(key)
        else:
            key = (sample, row["step"], row["layer_id"])
            all_key = ("decode", *key)
            require(row["phase"] == "decode", f"unknown router phase: {row['phase']}")
            require(row["token_id"] == row["input_token_id"], f"decode input token mismatch: {key}")
            require(key not in seen_decode, f"duplicate decode token/layer: {key}")
            seen_decode.add(key)
        require(all_key not in seen_all, f"duplicate router coordinate: {all_key}")
        seen_all.add(all_key)

    for row in decode_rows:
        sample, step, layer = row["sample_id"], row["step"], row["layer_id"]
        req = request_by_id[sample]
        outputs = req["parity"]["generated_token_ids"]
        require(row["token_id"] == outputs[step], f"decode trace token ID mismatch: {sample}/{step}")
        require(row["predicts_output_index"] == step + 1, f"decode output alignment mismatch: {sample}/{step}")
    decode_keys = {(r["sample_id"], r["step"], r["layer_id"], r["token_id"], r["predicts_output_index"])
                   for r in decode_rows}
    require(decode_keys == expected_decode, "decode trace coverage differs from output token sequence")
    prefill_keys = {(r["sample_id"], r["sequence_index"], r["layer_id"]) for r in prefill_rows}
    require(prefill_keys == seen_prefill, "prefill trace differs from full router calls")
    require(len(prefill_rows) == expected_prefill_count, "prefill token/layer row count mismatch")
    require(len(decode_rows) == len(expected_decode) == manifest["decode_rows"], "decode row count mismatch")
    require(decode_keys == {(r["sample_id"], r["step"], r["layer_id"], r["token_id"], r["predicts_output_index"])
                            for r in all_rows if r["phase"] == "decode"},
            "cache simulator input is not the exact decode projection")
    sorted_keys = [(r["sample_id"], r["step"], r["layer_id"]) for r in decode_rows]
    require(sorted_keys == sorted(sorted_keys), "decode trace is not sorted for simulator")
    require(len(all_rows) == len(seen_all) and len(seen_decode) + len(seen_prefill) == len(all_rows),
            "all-router-call coverage mismatch")
    for req, timing in zip(requests, timings["request_timings"], strict=True):
        require(req["sample_id"] == timing["sample_id"], "request timing order mismatch")
        require(abs(req["baseline_seconds"] - timing["baseline_seconds"]) < 1e-9
                and abs(req["instrumented_seconds"] - timing["instrumented_seconds"]) < 1e-9,
                f"request timings disagree: {req['sample_id']}")

    output_hashes = manifest["output_files"]
    for name, expected_hash in output_hashes.items():
        actual = hashlib.sha256((trace_dir / name).read_bytes()).hexdigest()
        require(actual == expected_hash, f"output file hash mismatch: {name}")
    smoke = json.loads((trace_dir / "smoke.json").read_bytes())
    smoke_parity = smoke["parity"]
    require(smoke_parity["token_ids_equal"] and smoke_parity["text_equal"]
            and smoke_parity["finish_reason_equal"], "smoke parity failed")
    smoke_rows = read_jsonl(trace_dir / "smoke_router_calls.jsonl")
    require(len(smoke_rows) == smoke["router_calls"], "smoke trace row count mismatch")
    require(any(r["phase"] == "decode" for r in smoke_rows), "smoke did not reach decode")

    result = {"status": "PASS", "formal_requests": len(requests),
              "prompt_tokens_by_request": {r["sample_id"]: r["prompt_tokens"] for r in requests},
              "output_tokens_by_request": {r["sample_id"]: len(r["parity"]["generated_token_ids"]) for r in requests},
              "decode_steps_by_request": {r["sample_id"]: r["decode_steps"] for r in requests},
              "prefill_rows": len(prefill_rows), "decode_rows": len(decode_rows),
              "layers": 26, "experts": 64, "top_k": 6,
              "cache_sim_schema_valid": True,
              "decode_step_semantics": manifest["decode_step_semantics"],
              "decode_trace_sha256": hashlib.sha256((trace_dir / "decode_trace.jsonl").read_bytes()).hexdigest()}
    (trace_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--workload", type=Path, required=True)
    args = parser.parse_args()
    verify(args.trace_dir, args.workload)


if __name__ == "__main__":
    main()
