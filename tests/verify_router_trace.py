"""Verify a saved M0-3 Router Trace with Python's standard library."""

import argparse
import json
import math
import struct
from itertools import product
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def softmax_matches(probability, expected, dtype):
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return False
    if dtype == "torch.float32":
        return abs(probability - expected) <= 2e-7
    if dtype != "torch.float16":
        return False
    encoded = struct.pack("<e", probability)
    rounded, = struct.unpack("<e", encoded)
    if probability != rounded:
        return False
    bits, = struct.unpack("<H", encoded)
    previous, = struct.unpack("<e", struct.pack("<H", max(0, bits - 1)))
    following, = struct.unpack("<e", struct.pack("<H", bits + 1))
    # Softmax internally uses float32 arithmetic. Allow its rounding error near
    # a float16 midpoint while still checking the observed half-precision value.
    slack = expected * 2e-7
    return (previous + probability) / 2 - slack <= expected <= (probability + following) / 2 + slack


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(manifest["status"] == "PASS", "trace run did not pass")
    stats = manifest["metrics"]
    input_ids = manifest["inputs"]["input_ids"]
    masks = manifest["inputs"]["attention_mask"]
    excluded = set(manifest["excluded_token_ids"])
    expected_tokens = {}
    for batch, (ids, mask) in enumerate(zip(input_ids, masks, strict=True)):
        for position, (token, present) in enumerate(zip(ids, mask, strict=True)):
            if present and token not in excluded:
                expected_tokens[(batch, position)] = token
    require(len(expected_tokens) == stats["valid_tokens"], "valid token count disagrees with inputs")
    expected_rows = set(product(stats["layer_ids"], expected_tokens))
    require(0 not in stats["layer_ids"], "dense layer must not appear in the routed trace")
    require(stats["logits_exact_equal"] and stats["logits_max_abs_difference"] == 0.0,
            "instrumented model logits differ")
    max_softmax_error = 0.0
    sums = []
    with (directory / manifest["trace"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require(row["evidence"] == manifest["evidence"], "record evidence disagrees with manifest")
            token_position = (row["batch_id"], row["sequence_index"])
            key = (row["layer_id"], token_position)
            require(key in expected_rows, f"unexpected or duplicate record {key}")
            expected_rows.remove(key)
            require(row["token_id"] == expected_tokens[token_position], f"wrong token ID for {key}")
            logits, probabilities = row["router_logits"], row["router_probabilities"]
            require(len(logits) == len(probabilities) == 64 and all(map(math.isfinite, logits)),
                    f"invalid router logits or probability length for {key}")
            maximum = max(logits)
            exponentials = [math.exp(value - maximum) for value in logits]
            denominator = sum(exponentials)
            dtype = row.get("router_softmax_dtype", "torch.float32")
            for probability, exp in zip(probabilities, exponentials, strict=True):
                expected = exp / denominator
                error = abs(probability - expected)
                require(softmax_matches(probability, expected, dtype), f"softmax mismatch for {key}")
                max_softmax_error = max(max_softmax_error, error)
            experts, weights = row["topk_expert_ids"], row["topk_weights"]
            require(len(experts) == len(weights) == len(set(experts)) == 6
                    and all(type(expert) is int and 0 <= expert < 64 for expert in experts),
                    f"invalid expert IDs or weight count for {key}")
            threshold = sorted(probabilities, reverse=True)[5]
            for expert, weight in zip(experts, weights, strict=True):
                require(math.isclose(weight, probabilities[expert], rel_tol=2e-6, abs_tol=2e-7)
                        and weight >= threshold - 2e-7,
                        f"selected weight disagrees with unnormalized Top-6 for {key}")
            sums.append(sum(weights))
    require(not expected_rows and len(sums) == stats["records"], "trace is incomplete")
    print(json.dumps({
        "evidence": manifest["evidence"], "records": len(sums),
        "valid_tokens": len(expected_tokens), "layer_ids": stats["layer_ids"],
        "max_probability_error_vs_python_softmax": max_softmax_error,
        "min_selected_weight_sum": min(sums), "max_selected_weight_sum": max(sums),
        "model_logits_max_abs_difference": stats["logits_max_abs_difference"],
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    verify(parser.parse_args().trace_dir)
