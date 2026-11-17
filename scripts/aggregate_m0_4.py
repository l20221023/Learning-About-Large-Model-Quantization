"""Validate and aggregate an M0-4 router trace using only Python's stdlib."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LAYERS = set(range(1, 27))
EXPERTS = set(range(64))
GROUPS = {"general", "chinese", "code"}


def load_rows(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("evidence") != "MEASURED_COMMUNITY_AWQ_INT4":
            raise ValueError(f"row {line_no}: unexpected evidence {row.get('evidence')!r}")
        if row.get("layer_id") not in LAYERS or row.get("workload_group") not in GROUPS:
            raise ValueError(f"row {line_no}: invalid layer/group")
        ids, weights, probs = row["topk_expert_ids"], row["topk_weights"], row["router_probabilities"]
        if len(ids) != 6 or len(set(ids)) != 6 or not set(ids) <= EXPERTS:
            raise ValueError(f"row {line_no}: invalid Top-6 IDs")
        if len(weights) != 6 or len(probs) != 64:
            raise ValueError(f"row {line_no}: invalid probability vector")
        if any(not math.isfinite(x) or x < 0 for x in probs + weights):
            raise ValueError(f"row {line_no}: non-finite or negative probability")
        if abs(sum(probs) - 1.0) > 0.005:
            raise ValueError(f"row {line_no}: router probabilities do not sum to one")
        rows.append(row)
    if not rows:
        raise ValueError("empty trace")
    seen = set()
    coordinate_layers = defaultdict(set)
    for row in rows:
        key = row["workload_group"], row["sample_id"], row["layer_id"], row["sequence_index"]
        if key in seen:
            raise ValueError(f"duplicate trace coordinate {key}")
        seen.add(key)
        coordinate_layers[key[0], key[1], key[3]].add(key[2])
    if any(layers != LAYERS for layers in coordinate_layers.values()):
        raise ValueError("a sample/token coordinate is missing one or more MoE layers")
    return rows


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, out, *, manifest=None):
    out.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(Counter)
    probability_sums = defaultdict(lambda: [0.0] * 64)
    coactivation = defaultdict(Counter)
    samples = defaultdict(set)
    sample_tokens = defaultdict(set)
    sample_counts = defaultdict(Counter)
    sample_prob_sums = defaultdict(lambda: [0.0] * 64)
    keys = set()
    for row in rows:
        group, sample, layer = row["workload_group"], row["sample_id"], row["layer_id"]
        token_key = group, sample, row["sequence_index"]
        keys.add(token_key)
        samples[group].add(sample)
        sample_tokens[group, sample].add(row["sequence_index"])
        group_layer = group, layer
        sample_layer = group, sample, layer
        for expert, prob in enumerate(row["router_probabilities"]):
            probability_sums[group_layer][expert] += prob
            sample_prob_sums[sample_layer][expert] += prob
        selected = row["topk_expert_ids"]
        counts[group_layer].update(selected)
        sample_counts[sample_layer].update(selected)
        selected_set = set(selected)
        for left in selected_set:
            for right in selected_set:
                coactivation[group_layer][left, right] += 1
    token_counts = Counter(group for group, _, _ in keys)
    expected_groups = {group for group in GROUPS if token_counts[group]}
    if expected_groups != GROUPS:
        raise ValueError(f"missing workload groups: {GROUPS - expected_groups}")
    if any((group, layer) not in counts for group in GROUPS for layer in LAYERS):
        raise ValueError("trace is missing a workload group/layer combination")
    routing_rows, coact_rows, load_rows, sample_count_rows, sample_load_rows = [], [], [], [], []
    selected_weight_sums = defaultdict(float)
    for row in rows:
        gl = row["workload_group"], row["layer_id"]
        for expert, weight in zip(row["topk_expert_ids"], row["topk_weights"]):
            selected_weight_sums[gl, expert] += weight
    for group in sorted(GROUPS):
        for layer in sorted(LAYERS):
            gl = group, layer
            n_tokens = token_counts[group]
            observed = sum(counts[gl].values())
            if observed != n_tokens * 6:
                raise ValueError(f"{group} layer {layer}: selections {observed} != tokens*6 ({n_tokens*6})")
            values = [counts[gl][e] for e in range(64)]
            mean, std = statistics.fmean(values), statistics.pstdev(values)
            load_rows.append({"workload_group": group, "layer_id": layer, "valid_tokens": n_tokens,
                              "selections": observed, "mean_expert_load": mean,
                              "load_cv": std / mean if mean else 0.0,
                              "max_to_mean": max(values) / mean if mean else 0.0})
            for expert in range(64):
                routing_rows.append({
                    "workload_group": group, "layer_id": layer, "expert_id": expert,
                    "selection_count": counts[gl][expert],
                    "mean_router_probability": probability_sums[gl][expert] / n_tokens,
                    "mean_selected_weight": selected_weight_sums[gl, expert] / counts[gl][expert]
                        if counts[gl][expert] else 0.0,
                })
            upper_pairs = 0
            for i in range(64):
                for j in range(64):
                    val = coactivation[gl][i, j]
                    if i != j and val != coactivation[gl][j, i]:
                        raise ValueError("coactivation matrix is not symmetric")
                    coact_rows.append({"workload_group": group, "layer_id": layer,
                                       "expert_i": i, "expert_j": j, "coactivation_count": val})
                    if i < j:
                        upper_pairs += val
            if upper_pairs != n_tokens * 15:
                raise ValueError(f"{group} layer {layer}: unique coactivation pairs {upper_pairs} != tokens*15")
    _write_csv(out / "routing_counts.csv", ["workload_group", "layer_id", "expert_id", "selection_count",
               "mean_router_probability", "mean_selected_weight"], routing_rows)
    _write_csv(out / "coactivation.csv", ["workload_group", "layer_id", "expert_i", "expert_j",
               "coactivation_count"], coact_rows)
    _write_csv(out / "load_summary.csv", list(load_rows[0]), load_rows)
    _write_csv(out / "workload_summary.csv", ["workload_group", "samples", "valid_tokens"],
               [{"workload_group": group, "samples": len(samples[group]), "valid_tokens": token_counts[group]}
                for group in sorted(GROUPS)])
    for group in sorted(GROUPS):
        for sample in sorted(samples[group]):
            n_tokens = len(sample_tokens[group, sample])
            for layer in sorted(LAYERS):
                sl = group, sample, layer
                vals = [sample_counts[sl][e] for e in range(64)]
                mean, std = statistics.fmean(vals), statistics.pstdev(vals)
                sample_load_rows.append({"workload_group": group, "sample_id": sample, "layer_id": layer,
                    "valid_tokens": n_tokens, "selections": sum(vals), "load_cv": std / mean if mean else 0.0,
                    "max_to_mean": max(vals) / mean if mean else 0.0})
                for expert in range(64):
                    sample_count_rows.append({"workload_group": group, "sample_id": sample,
                        "layer_id": layer, "expert_id": expert, "selection_count": sample_counts[sl][expert],
                        "mean_router_probability": sample_prob_sums[sl][expert] / n_tokens})
    _write_csv(out / "sample_routing_counts.csv", ["workload_group", "sample_id", "layer_id", "expert_id",
               "selection_count", "mean_router_probability"], sample_count_rows)
    _write_csv(out / "sample_load_summary.csv", list(sample_load_rows[0]), sample_load_rows)
    if manifest is not None:
        shared_rows = []
        shared_count = manifest.get("shared_expert_count_config")
        if shared_count != 2:
            raise ValueError("manifest does not identify the two shared experts")
        for sample in manifest["prompt_records"]:
            group = sample["workload_group"]
            calls = sample["shared_expert_calls"]
            if set(map(int, calls)) != LAYERS:
                raise ValueError(f"sample {sample['sample_id']}: shared layer coverage differs")
            for layer in sorted(LAYERS):
                call = calls[str(layer)]
                if call["calls"] != 1 or call["model_input_tokens"] != sample["prompt_tokens"]:
                    raise ValueError(f"sample {sample['sample_id']} layer {layer}: shared call count mismatch")
                shared_rows.append({
                    "workload_group": group, "sample_id": sample["sample_id"], "layer_id": layer,
                    "shared_expert_count_config": shared_count, "module": call["module"],
                    "module_calls": call["calls"], "model_input_tokens": call["model_input_tokens"],
                    "valid_content_tokens": sample["valid_tokens"],
                    "special_template_tokens": sample["prompt_tokens"] - sample["valid_tokens"],
                    "shared_expert_token_slots": shared_count * call["model_input_tokens"],
                })
        _write_csv(out / "shared_expert_activity.csv", list(shared_rows[0]), shared_rows)
    chart_dir = out / "layer_charts"
    chart_dir.mkdir()
    for layer in sorted(LAYERS):
        _write_layer_svg(chart_dir / f"layer_{layer:02}.svg", layer, counts, coactivation)
    return {"rows": len(rows), "groups": sorted(GROUPS), "valid_tokens_per_group": dict(token_counts),
            "layers": 26, "routed_experts": 64, "top_k": 6,
            "selection_invariant": "each group/layer selections = valid prompt tokens * 6",
            "coactivation_invariant": "each group/layer upper triangle sum = valid prompt tokens * 15"}


def _write_layer_svg(path, layer, counts, coact):
    width, height = 760, 560
    colors = {"general": "#377eb8", "chinese": "#4daf4a", "code": "#e41a1c"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-size="18">Layer {layer}: routed load by group</text>']
    for gi, group in enumerate(sorted(GROUPS)):
        y0 = 45 + gi * 90
        maximum = max(counts[group, layer].values(), default=1) or 1
        parts.append(f'<text x="8" y="{y0+16}" font-size="12">{group}</text>')
        for expert in range(64):
            h = 54 * counts[group, layer][expert] / maximum
            x = 75 + expert * 10
            parts.append(f'<rect x="{x}" y="{y0+58-h:.2f}" width="8" height="{h:.2f}" fill="{colors[group]}"/>')
    parts.append('<text x="20" y="333" font-size="18">Top-6 coactivation (all groups pooled)</text>')
    pooled = Counter()
    for group in GROUPS:
        pooled.update(coact[group, layer])
    peak = max(pooled.values(), default=1) or 1
    for i in range(64):
        for j in range(64):
            shade = round(255 * (1 - pooled[i, j] / peak))
            color = f"#{shade:02x}{shade:02x}ff"
            parts.append(f'<rect x="90" y="{350+i*3.1:.1f}" width="3.1" height="3.1" fill="{color}"/>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes()) if args.manifest else None
    result = aggregate(load_rows(args.trace), args.out, manifest=manifest)
    (args.out / "aggregate_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
