#!/usr/bin/env python3
"""Offline alignment and TP1/TP2 routed-expert comparison for P2-M3.

This script reads frozen artifacts only. It performs no model inference and does
not alter the source trace, O2 evidence, or shared project status.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M3 = ROOT / "stage_artifacts/P2-M3"
O2 = ROOT / "stage_artifacts/OffloadGate/BF16_decode_20261120T172330"
O2_TRACE = O2 / "remote-evidence/OffloadGate-BF16-decode-20261120T172330/trace"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise RuntimeError(f"invalid JSON {path}:{line_no}: {exc}") from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_ids_from_o2(req):
    return list(req["parity"]["baseline_token_ids"])


def output_ids_from_m3(record):
    return list(record["response"]["choices"][0]["token_ids"])


def position_bucket(phase: str, pos: int) -> str:
    if phase == "prefill":
        if pos < 8:
            return "prefill:0-7"
        if pos < 16:
            return "prefill:8-15"
        if pos < 32:
            return "prefill:16-31"
        if pos < 64:
            return "prefill:32-63"
        return "prefill:64+"
    if pos <= 8:
        return "decode:output-index-1-8"
    if pos <= 16:
        return "decode:output-index-9-16"
    if pos <= 24:
        return "decode:output-index-17-24"
    return "decode:output-index-25+"


def boundary_margin(logits):
    vals = sorted((float(x) for x in logits), reverse=True)
    if len(vals) != 64:
        return None
    return vals[5] - vals[6]


def summarize(values):
    xs = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    def q(p):
        return xs[min(len(xs) - 1, int(math.floor(p * (len(xs) - 1))))]
    return {
        "n": len(xs), "mean": sum(xs) / len(xs),
        "median": (xs[(len(xs)-1)//2] + xs[len(xs)//2]) / 2,
        "p10": q(0.10), "p90": q(0.90),
        "fraction_le_0_01": sum(x <= 0.01 for x in xs) / len(xs),
        "fraction_le_0_05": sum(x <= 0.05 for x in xs) / len(xs),
        "fraction_le_0_1": sum(x <= 0.1 for x in xs) / len(xs),
    }


def main():
    workload = json.loads((M3 / "workload.json").read_text(encoding="utf-8"))
    m3_work = {row["case_id"]: row for row in workload["rows"]}
    o2_requests = {r["sample_id"]: r for r in read_jsonl(O2_TRACE / "requests.jsonl")}
    m3_responses = {}
    for r in read_jsonl(M3 / "remote-runs/on-03/formal_responses.jsonl"):
        if r.get("phase") == "formal" and r.get("response"):
            m3_responses[r["case_id"].removeprefix("formal-")] = r

    mappings = []
    common_decode_steps = {}
    for case_id, row in sorted(m3_work.items()):
        o2 = o2_requests.get(case_id)
        m3_response = m3_responses.get(case_id)
        tp1_ids = list(o2["prompt_token_ids"]) if o2 else []
        tp2_ids = list(row["prompt_token_ids"])
        ids1 = output_ids_from_o2(o2) if o2 else []
        ids2 = output_ids_from_m3(m3_response) if m3_response else []
        common_n = min(len(ids1), len(ids2))
        first_difference = next((i for i, (a, b) in enumerate(zip(ids1, ids2)) if a != b), None)
        # A decode route with predicts_output_index=k consumes generated token
        # k-1. It is comparable only while the entire consumed history agrees.
        available_decode_steps = max(0, min(len(ids1), len(ids2)) - 1)
        common_decode_steps[case_id] = min(available_decode_steps, first_difference) if first_difference is not None else available_decode_steps
        mappings.append({
            "case_id": case_id,
            "tp1_sample_id": o2.get("sample_id") if o2 else None,
            "prompt_text_equal": bool(o2 and row["text"] == o2["prompt"]),
            "prompt_token_ids_equal": bool(o2 and tp1_ids == tp2_ids),
            "prompt_token_count_tp1": len(tp1_ids),
            "prompt_token_count_tp2": len(tp2_ids),
            "prompt_token_ids_sha256_tp1": hashlib.sha256(json.dumps(tp1_ids, separators=(",", ":")).encode()).hexdigest() if o2 else None,
            "prompt_token_ids_sha256_tp2": hashlib.sha256(json.dumps(tp2_ids, separators=(",", ":")).encode()).hexdigest(),
            "generated_token_count_tp1": len(ids1),
            "generated_token_count_tp2": len(ids2),
            "common_generated_prefix_tokens": common_n,
            "generated_prefix_equal": bool(common_n == len(ids2) and ids1[:common_n] == ids2[:common_n]),
            "generated_token_ids_first_difference": first_difference,
            "history_aligned_decode_steps": common_decode_steps[case_id],
            "o2_max_tokens": (o2.get("generation") or {}).get("max_tokens") if o2 else None,
        })

    # Build route records keyed by sample, phase, position, and layer. Prefill
    # uses token position. Decode uses predicts_output_index to align the
    # generated input token that produces the next output token.
    tp1 = {}
    tp1_dup = 0
    for r in read_jsonl(O2_TRACE / "router_calls.jsonl"):
        if r.get("request_kind") != "formal":
            continue
        sample = r["sample_id"]
        phase = r["phase"]
        pos = int(r["sequence_index"] if phase == "prefill" else r["predicts_output_index"])
        key = (sample, phase, pos, int(r["layer_id"]))
        if key in tp1:
            tp1_dup += 1
        tp1[key] = r

    tp2 = {}
    tp2_dup = 0
    for r in read_jsonl(M3 / "quality_trace.jsonl"):
        case = r["case_id"]
        phase = r["phase"]
        pos = int(r["token_row"] if phase == "prefill" else r["predicts_output_index"])
        key = (case, phase, pos, int(r["layer_id"]))
        if key in tp2:
            tp2_dup += 1
        tp2[key] = r

    comparisons = []
    missing = []
    excluded_decode = collections.Counter()
    excluded_decode_by_case = {}
    for key, a in tp1.items():
        sample, phase, pos, layer = key
        if phase == "decode":
            aligned_steps = common_decode_steps.get(sample, 0)
            if pos > aligned_steps:
                m3_len = len(output_ids_from_m3(m3_responses[sample])) if sample in m3_responses else 0
                if pos >= m3_len:
                    reason = "tp2_output_horizon_shorter"
                else:
                    reason = "generation_history_diverged"
                excluded_decode[reason] += 1
                excluded_decode_by_case.setdefault(sample, collections.Counter())[reason] += 1
                continue
        b = tp2.get(key)
        if b is None:
            missing.append({"key": list(key), "side_missing": "TP2"})
            continue
        a_tok = int(a.get("input_token_id", a.get("token_id")))
        b_tok = int(b["input_token_id"])
        if a_tok != b_tok:
            missing.append({"key": list(key), "side_missing": "token_mismatch", "tp1_token": a_tok, "tp2_token": b_tok})
            continue
        s1 = set(int(x) for x in a["topk_expert_ids"])
        s2 = set(int(x) for x in b["topk_expert_ids"])
        margin = boundary_margin(a.get("router_logits", []))
        comparisons.append({
            "key": key, "sample_id": key[0], "phase": key[1], "position": key[2], "layer_id": key[3],
            "token_id": a_tok, "set_equal": s1 == s2,
            "ordered_equal": list(a["topk_expert_ids"]) == list(b["topk_expert_ids"]),
            "intersection_count": len(s1 & s2), "symmetric_difference_count": len(s1 ^ s2),
            "tp1_boundary_logit_margin": margin,
        })

    by_phase = {}
    by_layer = {}
    by_phase_layer = {}
    by_position = {}
    for c in comparisons:
        phase = c["phase"]
        layer = str(c["layer_id"])
        bucket = position_bucket(phase, int(c["position"]))
        for d, k in ((by_phase, phase), (by_layer, layer), (by_phase_layer, f"{phase}:layer-{layer}"), (by_position, bucket)):
            z = d.setdefault(k, {"rows": 0, "set_mismatches": 0, "ordered_mismatches": 0, "margins_mismatch": [], "margins_match": []})
            z["rows"] += 1
            z["set_mismatches"] += int(not c["set_equal"])
            z["ordered_mismatches"] += int(not c["ordered_equal"])
            z["margins_mismatch" if not c["set_equal"] else "margins_match"].append(c["tp1_boundary_logit_margin"])

    def finish(d):
        out = {}
        for k, v in d.items():
            out[k] = {
                "rows": v["rows"], "set_mismatches": v["set_mismatches"],
                "set_mismatch_rate": v["set_mismatches"] / v["rows"] if v["rows"] else None,
                "ordered_mismatches": v["ordered_mismatches"],
                "tp1_boundary_margin_mismatch": summarize(v["margins_mismatch"]),
                "tp1_boundary_margin_match": summarize(v["margins_match"]),
            }
        return out

    comparable = [c for c in comparisons if c["phase"] == "prefill"]
    overall = {
        "rows": len(comparisons),
        "set_mismatches": sum(not c["set_equal"] for c in comparisons),
        "ordered_mismatches": sum(not c["ordered_equal"] for c in comparisons),
        "prefill_rows": sum(c["phase"] == "prefill" for c in comparisons),
        "prefill_set_mismatches": sum(c["phase"] == "prefill" and not c["set_equal"] for c in comparisons),
        "decode_rows": sum(c["phase"] == "decode" for c in comparisons),
        "decode_set_mismatches": sum(c["phase"] == "decode" and not c["set_equal"] for c in comparisons),
        "cross_side_missing_or_token_mismatch": len(missing),
        "excluded_decode_rows_by_reason": dict(excluded_decode),
        "excluded_decode_rows_by_case": {k: dict(v) for k, v in excluded_decode_by_case.items()},
        "duplicate_keys_tp1": tp1_dup,
        "duplicate_keys_tp2": tp2_dup,
        "tp1_boundary_margin_all": summarize([c["tp1_boundary_logit_margin"] for c in comparable]),
        "tp1_boundary_margin_mismatch": summarize([c["tp1_boundary_logit_margin"] for c in comparable if not c["set_equal"]]),
        "tp1_boundary_margin_match": summarize([c["tp1_boundary_logit_margin"] for c in comparable if c["set_equal"]]),
    }

    # Hash provenance for the exact frozen inputs and traces consumed.
    input_paths = [
        M3 / "workload.json", M3 / "quality_trace.jsonl",
        M3 / "remote-runs/on-03/formal_responses.jsonl",
        O2_TRACE / "requests.jsonl", O2_TRACE / "router_calls.jsonl",
        O2 / "cache_simulation.json", O2 / "independent_audit.json",
        M3 / "tp_control/tp_control_audit.json",
        M3 / "tp_control/remote/tp-control-tp1-20261120/trace/response.json",
        M3 / "tp_control/remote/tp-control-tp2-20261120/trace/response.json",
    ]
    result = {
        "analysis_status": "PASS" if not missing and all(x["prompt_token_ids_equal"] for x in mappings) else "PARTIAL",
        "causal_attribution_status": "PARTIAL",
        "scope": "offline-only; no inference; O2 TP1 and M3 TP2 historical runs are not configuration-matched",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in input_paths},
        "sample_mappings": mappings,
        "controlled_prefill_experiment": json.loads((M3 / "tp_control/tp_control_audit.json").read_text(encoding="utf-8")),
        "configuration_comparison": {
            "model_repo_revision_equal": True,
            "same_remote_local_snapshot_path_claimed": "/root/autodl-tmp/p2-deepseek-moe/models/DeepSeek-V2-Lite-Chat-BF16-85864749",
            "safetensors_shard_content_hashes_at_both_run_times": "not recorded; exact model-weight identity is not proven",
            "model_config_hash_tp1": "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d",
            "model_config_hash_tp2": "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d",
            "tokenizer_config_hash_tp1": "31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8",
            "tokenizer_config_hash_tp2": "31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8",
            "prompt_text_and_token_ids": "all six cases exactly equal; chat-template hash matches the frozen M3 workload",
            "generation": "both greedy temperature=0/top_p=1, but O2 max_tokens=32 seed=20261117 and M3 max_tokens=24 seed=42; first generated token histories differ in 3/6 prompts",
            "max_model_len": {"tp1": 256, "tp2": 2304, "equal": False},
            "max_num_seqs": {"tp1": 1, "tp2": 8, "equal": False},
            "prefix_cache": {"tp1": False, "tp2": True, "equal": False},
            "offload": {"tp1": "UVA, cpu_offload_gb=20", "tp2": "none recorded / standard resident weights", "equal": False},
            "tensor_parallel_size": {"tp1": 1, "tp2": 2, "the intended variable" : True},
            "dtype_and_vllm": {"tp1": "BF16, vLLM 0.26.0, V1", "tp2": "BF16, vLLM 0.26.0, V1", "equal_by_recorded_configuration": True},
            "prefix_cache_observation": "TP1 explicitly disabled; TP2 enabled and recorded 48 prefix-cache hit tokens across formal/cache probes; per-prompt cache-hit token mapping is unavailable",
        },
        "alignment": overall,
        "by_phase": finish(by_phase),
        "by_layer": finish(by_layer),
        "by_phase_layer": finish(by_phase_layer),
        "by_position_bucket": finish(by_position),
        "missing_or_token_mismatch_examples": missing[:100],
        "router_margin_note": "Only TP1 raw router_logits exist; TP2 logits and cross-TP logit deltas are unavailable. TP1 kth-vs-(k+1)th logit margin is descriptive evidence only, not proof that TP2 rank ordering crossed the boundary.",
    }
    out = M3 / "tp_routing_difference_analysis.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_status": result["analysis_status"], "alignment": overall, "mappings": mappings, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
