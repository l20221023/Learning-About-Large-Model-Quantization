"""Independently verify an M0-4 trace against its inputs and raw captures."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

LAYERS = set(range(1, 27))
EXPERTS = set(range(64))


def exact_softmax(logits):
    peak = max(logits)
    exp = [math.exp(float(value) - peak) for value in logits]
    total = math.fsum(exp)
    return [value / total for value in exp]


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def half_predecessor(value):
    bits, = struct.unpack("<H", struct.pack("<e", half(value)))
    return struct.unpack("<e", struct.pack("<H", max(0, bits - 1)))[0]


def half_softmax_matches(observed, expected):
    if not math.isfinite(observed) or not math.isfinite(expected) or not 0.0 <= observed <= 1.0:
        return False
    try:
        encoded = struct.pack("<e", observed)
    except (OverflowError, struct.error):
        return False
    rounded, = struct.unpack("<e", encoded)
    if rounded != observed:
        return False
    bits, = struct.unpack("<H", encoded)
    previous, = struct.unpack("<e", struct.pack("<H", max(0, bits - 1)))
    following, = struct.unpack("<e", struct.pack("<H", bits + 1))
    slack = expected * 2e-7
    return (previous + observed) / 2 - slack <= expected <= (observed + following) / 2 + slack


def verify(trace_dir, workload_path):
    manifest = json.loads((trace_dir / "manifest.json").read_bytes())
    workload_bytes = workload_path.read_bytes()
    workload = json.loads(workload_bytes)
    require(manifest["status"] == "PASS" and manifest["exit_code"] == 0, "run did not pass")
    require(manifest["model"] == "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ", "unexpected model")
    require(manifest["revision"] == "e9fdf80ec6f1d200a029c41004bee40993763b43", "unexpected revision")
    require(manifest["evidence"] == "MEASURED_COMMUNITY_AWQ_INT4", "unexpected evidence class")
    require(manifest["workload_sha256"] == hashlib.sha256(workload_bytes).hexdigest(), "workload hash mismatch")
    require(manifest["workload_id"] == workload["workload_id"], "workload ID mismatch")
    require(manifest["phase"] == "prefill", "unexpected phase")
    require(manifest["layer_ids"] == sorted(LAYERS), "missing routed layers")
    require(manifest["shared_expert_count_config"] == 2, "checkpoint does not declare two shared experts")
    special = set(manifest["special_token_ids"])
    samples = {}
    for record in manifest["prompt_records"]:
        sample, group = record["sample_id"], record["workload_group"]
        require(sample not in samples, f"duplicate sample metadata: {sample}")
        require(record["logits_max_abs_difference"] == 0.0, f"logits differ for {sample}")
        require(len(record["generated_token_ids"]) == 1, f"expected one decode token for {sample}")
        require(record["prompt_sha256"] == hashlib.sha256(record["prompt"].encode()).hexdigest(),
                f"prompt hash mismatch: {sample}")
        valid_positions = {i: token for i, token in enumerate(record["prompt_ids"]) if token not in special}
        require(len(valid_positions) == record["valid_tokens"], f"valid token count mismatch: {sample}")
        shared = record["shared_expert_calls"]
        require(set(map(int, shared)) == LAYERS, f"shared expert layer coverage differs: {sample}")
        require(all(call["calls"] == 1 and call["model_input_tokens"] == record["prompt_tokens"]
                    for call in shared.values()), f"shared expert activation count differs: {sample}")
        samples[sample] = (group, valid_positions, record)
    require(len(samples) == sum(map(len, workload["groups"].values())), "sample count differs from workload")
    require({group for group, _, _ in samples.values()} == set(workload["groups"]), "workload groups differ")
    raw_captures = {}
    for sample in samples:
        raw = json.loads((trace_dir / sample / "raw_route_capture.json").read_bytes())
        require(set(map(int, raw["layers"])) == LAYERS, f"raw layer coverage differs: {sample}")
        raw_captures[sample] = raw
    expected = {(sample, layer, position): token
                for sample, (group, positions, record) in samples.items()
                for layer in LAYERS for position, token in positions.items()}
    seen = set()
    max_softmax_error = 0.0
    max_weight_error = 0.0
    path = trace_dir / manifest["trace"]
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            sample = row["sample_id"]
            key = sample, row["layer_id"], row["sequence_index"]
            require(row["evidence"] == manifest["evidence"], f"evidence mismatch at row {line_no}")
            require(key in expected and key not in seen, f"unexpected or duplicate coordinate {key}")
            seen.add(key)
            require(row["workload_group"] == samples[sample][0], f"group mismatch for {key}")
            require(row["batch_id"] == 0 and row["token_id"] == expected[key], f"token mismatch for {key}")
            logits, probs = row["router_logits"], row["router_probabilities"]
            ids, weights = row["topk_expert_ids"], row["topk_weights"]
            require(len(logits) == len(probs) == 64 and all(map(math.isfinite, logits)),
                    f"bad logits/probability width for {key}")
            require(row["router_softmax_dtype"] == "torch.float16", f"unexpected precision for {key}")
            exact = exact_softmax(logits)
            for observed, expected_probability in zip(probs, exact):
                require(half_softmax_matches(observed, expected_probability),
                        f"softmax mismatch for {key}")
                max_softmax_error = max(max_softmax_error, abs(observed - expected_probability))
            require(len(ids) == len(weights) == 6 and len(set(ids)) == 6 and set(ids) <= EXPERTS,
                    f"invalid native Top-6 for {key}")
            for expert, weight in zip(ids, weights):
                require(half_softmax_matches(weight, exact[expert]),
                        f"selected weight is outside FP16 rounding interval for {key}")
                max_weight_error = max(max_weight_error, abs(weight - exact[expert]))
            top6 = set(sorted(EXPERTS, key=lambda expert: logits[expert], reverse=True)[:6])
            if set(ids) != top6:
                threshold = sorted(exact, reverse=True)[5]
                require(min(probs[expert] for expert in ids) >= half_predecessor(threshold),
                        f"native IDs fall below greedy Top-6 beyond FP16 tie rounding for {key}")
            raw = raw_captures[sample]
            dtype, raw_logits, raw_ids, raw_weights = raw["layers"][str(row["layer_id"])]
            pos = row["sequence_index"]
            require(raw["evidence"] == manifest["evidence"] and dtype == row["router_softmax_dtype"],
                    f"raw capture identity mismatch for {key}")
            require(raw_logits[pos] == logits and raw_ids[pos] == ids and raw_weights[pos] == weights,
                    f"trace differs from raw native capture for {key}")
    require(seen == set(expected), f"trace incomplete: missing {len(set(expected)-seen)} rows")
    require(len(seen) == manifest["record_count"], "manifest record count mismatch")
    require(all(record["trace_records"] == record["valid_tokens"] * 26 for _, _, record in samples.values()),
            "per-sample layer/token count invariant failed")
    print(json.dumps({"status": "PASS", "samples": len(samples), "groups": sorted(workload["groups"]),
                      "valid_tokens": {group: sum(len(positions) for g, positions, _ in samples.values() if g == group)
                                       for group in sorted(workload["groups"])},
                      "records": len(seen), "layers": 26, "experts": 64, "top_k": 6,
                      "max_probability_abs_error": max_softmax_error,
                      "max_native_weight_abs_error": max_weight_error,
                      "exact_logits_matches": len(samples)}, sort_keys=True))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--workload", type=Path, required=True)
    args = parser.parse_args()
    verify(args.trace_dir, args.workload)


if __name__ == "__main__":
    main()
