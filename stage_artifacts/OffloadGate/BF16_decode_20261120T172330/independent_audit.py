#!/usr/bin/env python3
"""Independent CPU audit of captured BF16 router routes and cache simulation."""
import collections
import hashlib
import json
import math
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def route_audit(trace_dir):
    rows = [json.loads(line) for line in (trace_dir / "router_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 4550 + 4836, "router-call total differs from prefill + decode manifest")
    require({row["layer_id"] for row in rows} == set(range(1, 27)), "route rows omit a layer")
    require(all(row["route_validation_softmax_dtype"] == "torch.bfloat16" for row in rows),
            "mixed or unexpected softmax dtype")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["sample_id"], row["layer_id"], row["phase"], row["step"])].append(row)
    weight_error = probability_error = 0.0
    cross_version_differences = []
    for key, group in groups.items():
        group.sort(key=lambda row: row["sequence_index"] if row["phase"] == "prefill" else 0)
        logits = torch.tensor([row["router_logits"] for row in group], dtype=torch.float32).to(torch.bfloat16)
        ids = torch.tensor([row["topk_expert_ids"] for row in group], dtype=torch.long)
        weights = torch.tensor([row["topk_weights"] for row in group], dtype=torch.float32)
        saved_probs = torch.tensor([row["topk_probabilities"] for row in group], dtype=torch.float32)
        require(logits.shape == (len(group), 64) and ids.shape == weights.shape == saved_probs.shape == (len(group), 6),
                f"router tensor shape mismatch: {key}")
        # Recompute per original native call batch; CPU softmax rounding can vary by batch shape.
        probs = logits.softmax(dim=1, dtype=torch.bfloat16).float()
        selected_probs = probs.gather(1, ids)
        threshold = probs.topk(6, dim=1, sorted=True).values[:, -1:]
        require(torch.all(selected_probs >= threshold - 1e-7).item(), f"native Top-6 below cutoff: {key}")
        werr = (weights - selected_probs).abs().max().item()
        perr = (saved_probs - selected_probs).abs().max().item()
        weight_error = max(weight_error, werr)
        probability_error = max(probability_error, perr)
        ulp = (torch.nextafter(logits.softmax(dim=1, dtype=torch.bfloat16),
                               torch.full_like(logits, float("inf"))).float()
               - logits.softmax(dim=1, dtype=torch.bfloat16).float()).gather(1, ids)
        require(torch.equal(weights, saved_probs), f"native weights differ from recorded selected probabilities: {key}")
        require(torch.all((weights - selected_probs).abs() <= ulp + 1e-8).item(),
                f"selected weights exceed one BF16 ULP from CPU recomputation: {key}")
        for row_index, slot in ((weights - selected_probs).abs() > 2e-7).nonzero().tolist():
            native = torch.tensor(weights[row_index, slot].item(), dtype=torch.bfloat16)
            native_f = float(native)
            adjacent = float(torch.nextafter(native, torch.tensor(float("-inf"), dtype=torch.bfloat16)))
            midpoint = (native_f + adjacent) / 2
            reference = logits[row_index].double().softmax(dim=0)[int(ids[row_index, slot])].item()
            cross_version_differences.append({
                "sample_id": group[row_index]["sample_id"], "phase": group[row_index]["phase"],
                "layer_id": group[row_index]["layer_id"],
                "sequence_index": group[row_index].get("sequence_index"),
                "step": group[row_index].get("step"), "expert_id": int(ids[row_index, slot]),
                "native_weight": native_f, "local_cpu_bf16_recompute": float(selected_probs[row_index, slot]),
                "absolute_difference": float((weights[row_index, slot] - selected_probs[row_index, slot]).abs()),
                "high_precision_softmax": reference, "adjacent_bf16_midpoint": midpoint,
                "distance_to_midpoint": abs(reference - midpoint),
            })
        require((ids >= 0).all().item() and (ids < 64).all().item(), f"expert ID outside 0..63: {key}")
        require((torch.sort(ids, dim=1).values[:, 1:] != torch.sort(ids, dim=1).values[:, :-1]).all().item(),
                f"duplicate selected expert IDs: {key}")
    require(len(cross_version_differences) == 1,
            f"expected the single documented BF16 midpoint discrepancy, found {len(cross_version_differences)}")
    require(cross_version_differences[0]["distance_to_midpoint"] < 5e-10,
            "BF16 cross-version discrepancy is not at an adjacent-value rounding midpoint")
    return {"router_rows": len(rows), "router_call_batches": len(groups), "layers": 26, "experts": 64, "top_k": 6,
            "softmax_dtype": "bfloat16", "max_abs_weight_error": weight_error,
            "max_abs_saved_probability_error": probability_error,
            "cpu_softmax_cross_version_difference_count": len(cross_version_differences),
            "cpu_softmax_cross_version_differences": cross_version_differences,
            "cpu_softmax_cross_version_tolerance": "no more than one adjacent BF16 value; differences are reported explicitly",
            "top6_cutoff_membership_checked_with_ties": True}


def simulate_reference(rows, capacity, policy):
    caches, last, frequency = {}, {}, collections.Counter()
    per_layer = collections.defaultdict(lambda: [0, 0])
    hits = misses = 0
    for index, row in enumerate(rows):
        key = (row["sample_id"], row["layer_id"])
        requested = set(row["topk_expert_ids"])
        require(len(requested) == 6 and len(requested) <= capacity, "bad atomic Top-6 request")
        cache = caches.setdefault(key, set())
        row_hits = len(requested & cache)
        row_misses = 6 - row_hits
        hits += row_hits
        misses += row_misses
        per_layer[row["layer_id"]][0] += row_hits
        per_layer[row["layer_id"]][1] += row_misses
        for expert in sorted(requested - cache):
            if len(cache) == capacity:
                candidates = cache - requested
                if policy == "lru":
                    victim = min(candidates, key=lambda e: (last[(key[0], key[1], e)], e))
                else:
                    victim = min(candidates, key=lambda e: (frequency[(key[0], key[1], e)],
                                                             last[(key[0], key[1], e)], e))
                cache.remove(victim)
            cache.add(expert)
        for expert in requested:
            last[(key[0], key[1], expert)] = index
            frequency[(key[0], key[1], expert)] += 1
    return hits, misses, {str(k): {"hits": v[0], "misses": v[1]} for k, v in sorted(per_layer.items())}


def static_hot_reference(rows, capacity):
    samples = sorted({r["sample_id"] for r in rows})
    total_hits = total_misses = total_preload = 0
    per_layer = collections.defaultdict(lambda: {"hits": 0, "misses": 0, "preload": 0})
    for sample in samples:
        training = collections.defaultdict(collections.Counter)
        for row in rows:
            if row["sample_id"] != sample:
                training[row["layer_id"]].update(row["topk_expert_ids"])
        hot_by_layer = {layer: {expert for expert, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:capacity]}
                        for layer, counts in training.items()}
        for layer, hot in hot_by_layer.items():
            total_preload += len(hot)
            per_layer[layer]["preload"] += len(hot)
        for row in rows:
            if row["sample_id"] != sample:
                continue
            layer = row["layer_id"]
            requested = set(row["topk_expert_ids"])
            h = len(requested & hot_by_layer[layer])
            m = len(requested) - h
            total_hits += h
            total_misses += m
            per_layer[layer]["hits"] += h
            per_layer[layer]["misses"] += m
    return total_hits, total_misses, total_preload, {str(k): v for k, v in sorted(per_layer.items())}


def cache_audit(trace_dir, simulation_path, expected_expert_bytes=17_301_504):
    trace_path = trace_dir / "decode_trace.jsonl"
    raw = trace_path.read_bytes()
    trace_hash = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    sim = json.loads(simulation_path.read_text(encoding="utf-8"))
    require(sim["trace_sha256"] == trace_hash, "simulator input hash mismatch")
    require(sim["expert_bytes"] == expected_expert_bytes, "per-expert byte assumption mismatch")
    require(len(rows) == 4836, "unexpected decode input row count")
    require([(r["sample_id"], r["step"], r["layer_id"]) for r in rows] ==
            sorted((r["sample_id"], r["step"], r["layer_id"]) for r in rows), "decode rows unsorted")
    require(len({(r["sample_id"], r["step"], r["layer_id"]) for r in rows}) == len(rows),
            "duplicate decode request-step-layer")
    require(all(len(r["topk_expert_ids"]) == 6 and len(set(r["topk_expert_ids"])) == 6
                and all(0 <= e < 64 for e in r["topk_expert_ids"]) for r in rows), "invalid Top-6 simulator input")
    samples = {r["sample_id"] for r in rows}
    layer_counts = collections.Counter(r["layer_id"] for r in rows)
    access_distribution = collections.defaultdict(collections.Counter)
    for row in rows:
        access_distribution[row["layer_id"]].update(row["topk_expert_ids"])
    expected_distribution = {str(layer): {str(expert): count for expert, count in sorted(counts.items())}
                             for layer, counts in sorted(access_distribution.items())}
    require(sim["access_distribution"] == expected_distribution, "per-layer expert access distribution mismatch")
    results = sim["results"]
    require(len(results) == 18, "expected 18 policy/capacity result rows")
    checked = []
    for cap in (6, 8, 12, 16, 32, 64):
        lru, lfu = [r for r in results if r["capacity_experts_per_layer"] == cap
                    and r["policy"] in ("lru", "lfu")]
        for result, policy in ((lru, "lru"), (lfu, "lfu")):
            h, m, per_layer = simulate_reference(rows, cap, policy)
            require((result["hits"], result["misses"], result["per_layer"]) == (h, m, per_layer),
                    f"{policy} hit/miss mismatch at capacity {cap}")
            require(result["theoretical_h2d_bytes"] == m * expected_expert_bytes,
                    f"{policy} byte mismatch at capacity {cap}")
            checked.append({"policy": policy, "capacity": cap, "hits": h, "misses": m,
                            "hit_rate": h / (h + m), "theoretical_h2d_bytes": m * expected_expert_bytes,
                            "per_layer": per_layer})
        static = next(r for r in results if r["capacity_experts_per_layer"] == cap
                      and r["policy"] == "static_hot_leave_one_request_out")
        h, m, preload, per_layer = static_hot_reference(rows, cap)
        require((static["hits"], static["misses"], static["initial_loads"], static["per_layer"])
                == (h, m, preload, per_layer), f"static-hot mismatch at capacity {cap}")
        require(static["theoretical_h2d_bytes"] == (preload + m) * expected_expert_bytes,
                f"static-hot byte mismatch at capacity {cap}")
        checked.append({"policy": "static_hot_leave_one_request_out", "capacity": cap,
                        "hits": h, "misses": m, "initial_load_experts": preload,
                        "hit_rate": h / (h + m),
                        "theoretical_h2d_bytes": (preload + m) * expected_expert_bytes,
                        "per_layer": per_layer})
    return {"status": "PASS", "trace_sha256": trace_hash, "decode_rows": len(rows),
            "requests": len(samples), "layers": len(layer_counts), "access_distribution": expected_distribution,
            "policy_capacity_results": checked,
            "limitations": ["simulated independent per-layer caches reset per request",
                            "Top-6 sets treated as simultaneous and experts as equal-sized",
                            "static-hot trained on other requests; initial preload counted",
                            "transfer byte totals are theoretical and are not latency or PCIe measurements"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("simulation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"route_audit": route_audit(args.trace_dir),
              "cache_audit": cache_audit(args.trace_dir, args.simulation)}
    result["status"] = "PASS"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "route_rows": result["route_audit"]["router_rows"],
                      "simulator_result_rows": len(result["cache_audit"]["policy_capacity_results"]),
                      "trace_sha256": result["cache_audit"]["trace_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
