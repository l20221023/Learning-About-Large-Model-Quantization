"""Validate and summarize same-token routed-expert shadow-forward outputs."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty M0-5 result rows")
    seen = set()
    by_token = defaultdict(set)
    experts = set()
    for row in rows:
        key = row["sample_id"], row["token_position"], row["expert_id"]
        if key in seen:
            raise ValueError(f"duplicate sample/token/expert coordinate: {key}")
        seen.add(key)
        by_token[row["sample_id"], row["token_position"]].add(row["expert_id"])
        experts.add(row["expert_id"])
        if row["layer_id"] != 13 or row["evidence"] != "MEASURED_COMMUNITY_AWQ_INT4":
            raise ValueError("unexpected layer or evidence class")
        if row["frequency_class"] not in {"high", "low"}:
            raise ValueError("invalid M0-4 frequency class")
        for name in ("cosine_similarity", "relative_l2", "left_norm", "right_norm",
                     "norm_ratio_right_over_left", "absolute_l2", "native_router_weight",
                     "fixed_weight_replacement_relative_l2"):
            if not math.isfinite(row[name]):
                raise ValueError(f"non-finite metric: {name}")
        if not -1.000001 <= row["cosine_similarity"] <= 1.000001:
            raise ValueError("cosine similarity outside [-1, 1]")
        if row["left_norm"] < 0 or row["right_norm"] < 0 or row["native_router_weight"] < 0:
            raise ValueError("negative norm or router weight")
        if row["shadow_reconstruction_allclose"] is not True:
            raise ValueError("shadow reconstruction gate did not pass")
        if not math.isfinite(row["shadow_reconstruction_relative_l2"]):
            raise ValueError("non-finite reconstruction error")
    if any(ids != experts for ids in by_token.values()):
        raise ValueError("candidate experts do not cover every sample/token")
    return rows


def load_pairs(path, rows):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("empty expert pair metrics")
    expected_experts = {r["expert_id"] for r in rows}
    expected = {tuple(sorted((a, b))) for i, a in enumerate(sorted(expected_experts))
                for b in sorted(expected_experts)[i + 1:]}
    seen = set()
    for row in records:
        pair = tuple(sorted((row["expert_a"], row["expert_b"])))
        key = row["sample_id"], row["token_position"], pair
        if pair not in expected or key in seen:
            raise ValueError(f"invalid or duplicate expert pair row: {key}")
        seen.add(key)
        if row["evidence"] != "MEASURED_COMMUNITY_AWQ_INT4" or row["layer_id"] != 13:
            raise ValueError("unexpected pair evidence or layer")
        for name in ("cosine_similarity", "relative_l2", "left_norm", "right_norm",
                     "norm_ratio_right_over_left", "absolute_l2"):
            if not math.isfinite(row[name]):
                raise ValueError(f"non-finite pair metric: {name}")
    expected_keys = {(r["sample_id"], r["token_position"], pair)
                     for r in rows for pair in expected}
    if seen != expected_keys:
        raise ValueError("pair metrics do not cover every sample/token/expert pair")
    return records


def aggregate(rows, pairs, out):
    out.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["expert_id"], row["frequency_class"]].append(row)
    fields = ["expert_id", "frequency_class", "observations", "mean_cosine_similarity",
              "mean_relative_l2", "mean_norm_ratio_right_over_left", "mean_native_router_weight",
              "native_selected_observations", "mean_fixed_weight_replacement_relative_l2"]
    with (out / "expert_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (expert, cls), values in sorted(grouped.items()):
            n = len(values)
            writer.writerow({
                "expert_id": expert, "frequency_class": cls, "observations": n,
                "mean_cosine_similarity": sum(r["cosine_similarity"] for r in values) / n,
                "mean_relative_l2": sum(r["relative_l2"] for r in values) / n,
                "mean_norm_ratio_right_over_left": sum(r["norm_ratio_right_over_left"] for r in values) / n,
                "mean_native_router_weight": sum(r["native_router_weight"] for r in values) / n,
                "native_selected_observations": sum(bool(r["native_selected"]) for r in values),
                "mean_fixed_weight_replacement_relative_l2": sum(
                    r["fixed_weight_replacement_relative_l2"] for r in values) / n,
            })
    pair_groups = defaultdict(list)
    for row in pairs:
        pair_groups[row["expert_a"], row["expert_b"]].append(row)
    with (out / "pair_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["expert_a", "expert_b", "observations", "mean_cosine_similarity",
                  "mean_relative_l2", "mean_norm_ratio_b_over_a"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (left, right), values in sorted(pair_groups.items()):
            count = len(values)
            writer.writerow({"expert_a": left, "expert_b": right, "observations": count,
                             "mean_cosine_similarity": sum(r["cosine_similarity"] for r in values) / count,
                             "mean_relative_l2": sum(r["relative_l2"] for r in values) / count,
                             "mean_norm_ratio_b_over_a": sum(r["norm_ratio_right_over_left"]
                                                              for r in values) / count})
    (out / "manifest.json").write_text(json.dumps({
        "status": "PASS", "row_count": len(rows), "layer_id": 13,
        "groups": sorted({r["workload_group"] for r in rows}),
        "samples": sorted({r["sample_id"] for r in rows}),
        "token_positions": sorted({r["token_position"] for r in rows}),
        "frequency_classes": {r["expert_id"]: r["frequency_class"] for r in rows},
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.input)
    aggregate(rows, load_pairs(args.pairs, rows), args.out)


if __name__ == "__main__":
    main()
