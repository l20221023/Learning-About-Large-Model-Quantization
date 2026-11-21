#!/usr/bin/env python3
"""Independent audit of captured routes, parity, mappings and paired metrics."""
import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--evidence-root", default="/root/autodl-tmp/experiments/P2-M3")
args = parser.parse_args()
evidence = Path(args.evidence_root)
workload = json.loads((ROOT / "workload.json").read_text(encoding="utf-8"))
rows_by_case = {x["case_id"]: x for x in workload["rows"]}
errors = []
checks = {}


def read_records(mode, phase, run_tag):
    path = ROOT / "remote-runs" / run_tag / f"{phase}_responses.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["case_id"]: r for r in records}


def fail(message):
    errors.append(message)


on_smoke = read_records("on", "smoke", "on-03")
off_smoke = read_records("off", "smoke", "off-01")
on = read_records("on", "formal", "on-03")
off = read_records("off", "formal", "off-01")
expected_formal = {f"formal-{case}" for case in rows_by_case}
expected_formal |= {"cache-probe-1", "cache-probe-2"}
checks["fixed_workload_count"] = len(rows_by_case)
if set(on) != expected_formal:
    fail(f"ON formal case IDs mismatch: {sorted(set(on) ^ expected_formal)}")
if set(off) != expected_formal:
    fail(f"OFF formal case IDs mismatch: {sorted(set(off) ^ expected_formal)}")
if set(on_smoke) != {"smoke-general-01"} or set(off_smoke) != {"smoke-general-01"}:
    fail("smoke case ID set mismatch")


def response(record):
    return record.get("response") or {}


def choice(record):
    choices = response(record).get("choices") or []
    if len(choices) != 1:
        fail(f"{record.get('case_id')}: expected one response choice, got {len(choices)}")
        return {}
    return choices[0]


def numeric_metrics(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s|$)", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def metric_delta(records, base, suffix):
    first = records[base[0]]["metrics_before"]
    last = records[base[1]]["metrics_after"]
    before, after = numeric_metrics(first), numeric_metrics(last)
    needle = suffix
    keys = {k for k in before.keys() | after.keys() if needle in k}
    sums = [k for k in keys if k.endswith("_sum")]
    counts = [k for k in keys if k.endswith("_count")]
    if not sums or not counts:
        return None
    return {
        "sum_seconds": sum(after.get(k, 0.0) - before.get(k, 0.0) for k in sums),
        "count": int(round(sum(after.get(k, 0.0) - before.get(k, 0.0) for k in counts))),
        "metric_names": sorted(sums + counts),
    }


def parity_pair(a, b, label):
    ca, cb = choice(a), choice(b)
    if not a.get("error") is None or not b.get("error") is None or not response(a) or not response(b):
        fail(f"{label}: request/API failure ON={a.get('error')} OFF={b.get('error')}")
        return None
    aid, bid = ca.get("token_ids"), cb.get("token_ids")
    if not isinstance(aid, list) or aid != bid:
        fail(f"{label}: output token IDs differ or are missing")
    lpa = (ca.get("logprobs") or {}).get("top_logprobs") or []
    lpb = (cb.get("logprobs") or {}).get("top_logprobs") or []
    if not lpa or len(lpa) != len(lpb):
        fail(f"{label}: top-5 logprobs missing or length mismatch")
        max_diff = None
    else:
        max_diff = 0.0
        for idx, (xa, xb) in enumerate(zip(lpa, lpb)):
            if set(xa) != set(xb):
                fail(f"{label}: top-5 logprob token set differs at output index {idx}")
                continue
            for token in xa:
                max_diff = max(max_diff, abs(float(xa[token]) - float(xb[token])))
        if max_diff > 1e-5:
            fail(f"{label}: top-5 logprob max abs difference {max_diff} exceeds 1e-5")
    if ca.get("text") != cb.get("text"):
        fail(f"{label}: decoded text differs despite token comparison")
    return {"output_tokens": len(aid) if isinstance(aid, list) else None,
            "top5_logprob_max_abs_diff": max_diff,
            "text_equal": ca.get("text") == cb.get("text")}


parity = {}
for case in sorted(expected_formal):
    parity[case] = parity_pair(on[case], off[case], case)
parity["smoke-general-01"] = parity_pair(on_smoke["smoke-general-01"], off_smoke["smoke-general-01"], "smoke-general-01")
checks["parity_cases"] = len(parity)
for case in expected_formal | {"smoke-general-01"}:
    a = on[case] if case in on else on_smoke[case]
    b = off[case] if case in off else off_smoke[case]
    if a.get("request_payload") != b.get("request_payload"):
        fail(f"{case}: ON/OFF request payloads differ")
all_ids = [response(r).get("id") for r in list(on.values()) + list(off.values()) + list(on_smoke.values()) + list(off_smoke.values())]
if len(set(all_ids)) != len(all_ids):
    fail("duplicate public API response IDs across collected requests")


def load_and_check_trace(record, run_tag, label):
    resp = response(record)
    c = choice(record)
    payload_ids = record["request_payload"]["prompt"]
    output_ids = c.get("token_ids") or []
    usage = resp.get("usage") or {}
    if usage.get("prompt_tokens") != len(payload_ids):
        fail(f"{label}: API prompt usage differs from frozen explicit token IDs")
    returned_prompt_ids = c.get("prompt_token_ids")
    if returned_prompt_ids is not None and list(map(int, returned_prompt_ids)) != list(map(int, payload_ids)):
        fail(f"{label}: response prompt_token_ids differ from the explicit request IDs")
    if usage.get("completion_tokens") != len(output_ids):
        fail(f"{label}: API completion usage differs from returned token_ids")
    arr_path = record.get("trace_array_path")
    encoded = c.get("routed_experts")
    if not arr_path or not encoded:
        fail(f"{label}: built-in routed_experts missing")
        return None
    arr_saved = np.load(ROOT / "remote-runs" / run_tag / arr_path, allow_pickle=False)
    arr_raw = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
    if not np.array_equal(arr_saved, arr_raw):
        fail(f"{label}: saved .npy differs from API raw base64 array")
    expected_rows = len(payload_ids) + len(output_ids) - 1
    if arr_raw.shape != (expected_rows, 27, 6):
        fail(f"{label}: expected route array {(expected_rows, 27, 6)}, got {arr_raw.shape}")
        return arr_raw
    if not np.issubdtype(arr_raw.dtype, np.integer):
        fail(f"{label}: route dtype is not integral: {arr_raw.dtype}")
    if np.any(arr_raw[:, 0, :] != 0):
        fail(f"{label}: dense layer 0 slot contains nonzero IDs")
    moe = arr_raw[:, 1:27, :]
    if moe.min() < 0 or moe.max() > 63:
        fail(f"{label}: routed ID out of [0,63], min={moe.min()} max={moe.max()}")
    if any(len(set(map(int, x))) != 6 for x in moe.reshape(-1, 6)):
        fail(f"{label}: Top-6 IDs are not unique within a token/layer")
    return arr_raw


route_arrays = {}
trace_out = ROOT / "quality_trace.jsonl"
cache_out = ROOT / "prefix_cache_trace.jsonl"
with trace_out.open("w", encoding="utf-8") as ftrace, cache_out.open("w", encoding="utf-8") as fcache:
    for case_id, record in on.items():
        arr = load_and_check_trace(record, "on-03", case_id)
        if arr is None:
            continue
        route_arrays[case_id] = arr
        if not case_id.startswith("formal-") and not case_id.startswith("cache-probe-"):
            continue
        sample_id = case_id.removeprefix("formal-") if case_id.startswith("formal-") else case_id
        source = rows_by_case.get(sample_id)
        prompt_ids = record["request_payload"]["prompt"]
        output_ids = choice(record).get("token_ids") or []
        target = fcache if case_id.startswith("cache-probe-") else ftrace
        for token_row in range(arr.shape[0]):
            if token_row < len(prompt_ids):
                phase = "prefill"
                token_id = int(prompt_ids[token_row])
                predicts_output_index = 0
                completion_input_index = None
            else:
                phase = "decode"
                completion_input_index = token_row - len(prompt_ids)
                token_id = int(output_ids[completion_input_index])
                predicts_output_index = completion_input_index + 1
            for layer_id in range(1, 27):
                item = {
                    "evidence": "MEASURED_VLLM_0260_TP2_BUILTIN_CAPTURE",
                    "case_id": sample_id,
                    "api_response_id": response(record).get("id"),
                    "internal_scheduler_request_id": None,
                    "phase": phase,
                    "token_row": token_row,
                    "input_token_id": token_id,
                    "completion_input_index": completion_input_index,
                    "predicts_output_index": predicts_output_index,
                    "layer_id": layer_id,
                    "topk_expert_ids": [int(x) for x in arr[token_row, layer_id, :]],
                }
                target.write(json.dumps(item, ensure_ascii=False) + "\n")

smoke_array = load_and_check_trace(on_smoke["smoke-general-01"], "on-03", "smoke-general-01")
if smoke_array is not None:
    case = rows_by_case["general-01"]
    if smoke_array.shape[0] != len(case["prompt_token_ids"]) + len(choice(on_smoke["smoke-general-01"]).get("token_ids", [])) - 1:
        fail("smoke: route/token row mapping mismatch")

checks["on_trace_cases"] = len(route_arrays)
checks["formal_trace_cases"] = sum(k.startswith("formal-") for k in route_arrays)
formal_input_rows = sum(route_arrays[f"formal-{case}"].shape[0] for case in rows_by_case if f"formal-{case}" in route_arrays)
formal_route_rows_expected = formal_input_rows * 26
formal_expert_choices_expected = formal_route_rows_expected * 6
quality_trace_lines = sum(1 for _ in trace_out.open(encoding="utf-8"))
if quality_trace_lines != formal_route_rows_expected:
    fail(f"normalized formal trace row count mismatch: {quality_trace_lines} != {formal_route_rows_expected}")
checks["formal_processed_token_rows"] = formal_input_rows
checks["formal_layer_token_rows"] = formal_route_rows_expected
checks["formal_expert_selections"] = formal_expert_choices_expected
checks["normalized_trace_jsonl_rows"] = quality_trace_lines

# Cache reuse must return the same logical prompt routes, even though the prompt
# was served earlier and its KV prefix is eligible for reuse.
cache_ok = True
base = route_arrays.get("formal-general-01")
for probe in ("cache-probe-1", "cache-probe-2"):
    arr = route_arrays.get(probe)
    if base is None or arr is None or not np.array_equal(base[:len(rows_by_case["general-01"]["prompt_token_ids"])], arr[:len(rows_by_case["general-01"]["prompt_token_ids"]) ]):
        cache_ok = False
        fail(f"{probe}: prompt route rows differ from the same prompt's first formal trace")
checks["prefix_cache_prompt_route_reuse_equal"] = cache_ok

def prefix_hits_delta(records, start_case, end_case):
    before = numeric_metrics(records[start_case]["metrics_before"])
    after = numeric_metrics(records[end_case]["metrics_after"])
    key = "vllm:prefix_cache_hits_total"
    return int(round(after.get(key, 0.0) - before.get(key, 0.0)))

prefix_cache_hits = {
    "on_formal_and_cache_probe_window": prefix_hits_delta(on, "formal-general-01", "cache-probe-2"),
    "off_formal_and_cache_probe_window": prefix_hits_delta(off, "formal-general-01", "cache-probe-2"),
}
if prefix_cache_hits["on_formal_and_cache_probe_window"] <= 0 or prefix_cache_hits["off_formal_and_cache_probe_window"] <= 0:
    fail(f"prefix-cache hit counter did not increase in both modes: {prefix_cache_hits}")
checks["prefix_cache_hit_tokens"] = prefix_cache_hits

# Compare all prompt-token routes to the independently captured native Top-6
# TP=1 BF16 trace for the exact same six frozen token sequences.
ref_path = Path("/root/autodl-tmp/experiments/OffloadGate-BF16-decode-20261120T172330/trace/prefill_trace.jsonl")
ref_map = {}
ref_token_map = {}
if ref_path.exists():
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("phase") == "prefill":
            key = (item["sample_id"], int(item["sequence_index"]), int(item["layer_id"]))
            ref_map[key] = item["topk_expert_ids"]
            ref_token_map[key] = int(item.get("input_token_id", item.get("token_id")))
else:
    fail(f"TP1 comparison reference missing: {ref_path}")
tp_compared = 0
tp_mismatches = 0
tp_token_mismatches = 0
for case_id, source in rows_by_case.items():
    arr = route_arrays.get("formal-" + case_id)
    if arr is None:
        continue
    for pos, token_id in enumerate(source["prompt_token_ids"]):
        reference_token = ref_token_map.get((case_id, pos, 1))
        if reference_token is None or reference_token != int(token_id):
            tp_token_mismatches += 1
        for layer_id in range(1, 27):
            key = (case_id, pos, layer_id)
            ref = ref_map.get(key)
            tp_compared += 1
            if ref is None or list(map(int, ref)) != list(map(int, arr[pos, layer_id, :])):
                tp_mismatches += 1
if tp_compared == 0 or tp_token_mismatches:
    fail(f"TP1 reference prompt token mapping mismatches={tp_token_mismatches}")
checks["tp1_reference_prompt_rows_compared"] = tp_compared
checks["tp1_reference_prompt_route_mismatches"] = tp_mismatches
checks["tp1_reference_prompt_token_mismatches"] = tp_token_mismatches

# Pair-level request metrics: deltas exclude server startup and smoke by using
# the before snapshot of formal request 1 and after snapshot of formal request 6.
performance = {}
for mode, records in (("on", on), ("off", off)):
    formal_ids = [f"formal-{case}" for case in rows_by_case]
    ordered = [records[k] for k in formal_ids if k in records]
    if len(ordered) == 6:
        met = {k: {"metrics_before": x["metrics_before"], "metrics_after": x["metrics_after"]} for k, x in zip(formal_ids, ordered)}
        ttft = metric_delta(met, (formal_ids[0], formal_ids[-1]), "time_to_first_token_seconds")
        tpot = metric_delta(met, (formal_ids[0], formal_ids[-1]), "request_time_per_output_token_seconds")
        wall_start = min(x["start_perf_counter_ns"] for x in ordered)
        wall_end = max(x["end_perf_counter_ns"] for x in ordered)
        output_tokens = sum(len(choice(x).get("token_ids") or []) for x in ordered)
        client_ms = [float(x["client_elapsed_ms"]) for x in ordered]
        performance[mode] = {
            "formal_successes": sum(bool(response(x)) and x.get("error") is None for x in ordered),
            "formal_requests": len(ordered),
            "output_tokens": output_tokens,
            "client_request_ms_mean": sum(client_ms) / len(client_ms),
            "client_request_ms_min": min(client_ms),
            "client_request_ms_max": max(client_ms),
            "aggregate_client_throughput_tokens_per_second": output_tokens / ((wall_end-wall_start)/1e9),
            "vllm_ttft_histogram_delta": ttft,
            "vllm_request_tpot_histogram_delta": tpot,
            "vllm_ttft_mean_seconds": (ttft["sum_seconds"] / ttft["count"] if ttft and ttft["count"] else None),
            "vllm_request_tpot_mean_seconds": (tpot["sum_seconds"] / tpot["count"] if tpot and tpot["count"] else None),
        }
    else:
        fail(f"{mode}: insufficient formal requests for metrics")

def relative_delta(on_value, off_value):
    if on_value is None or off_value in (None, 0):
        return None
    return {"absolute": on_value - off_value, "relative_percent": (on_value / off_value - 1.0) * 100.0}

capture_overhead = {
    "mean_client_request_ms": relative_delta(performance.get("on", {}).get("client_request_ms_mean"), performance.get("off", {}).get("client_request_ms_mean")),
    "mean_vllm_ttft_seconds": relative_delta(performance.get("on", {}).get("vllm_ttft_mean_seconds"), performance.get("off", {}).get("vllm_ttft_mean_seconds")),
    "mean_vllm_tpot_seconds": relative_delta(performance.get("on", {}).get("vllm_request_tpot_mean_seconds"), performance.get("off", {}).get("vllm_request_tpot_mean_seconds")),
    "aggregate_client_throughput_tokens_per_second": relative_delta(performance.get("on", {}).get("aggregate_client_throughput_tokens_per_second"), performance.get("off", {}).get("aggregate_client_throughput_tokens_per_second")),
}


def memory_stats(mode, run_slot):
    path = evidence / f"{mode}-{run_slot}"
    gpu = []
    gp = path / "gpu-sample.csv"
    if gp.exists():
        for line in gp.read_text(errors="ignore").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                try: gpu.append((int(parts[1]), int(parts[2])))
                except ValueError: pass
    rss = []
    cp = path / "cpu-rss-sample.csv"
    if cp.exists():
        for line in cp.read_text(errors="ignore").splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                try: rss.append(int(parts[1]))
                except ValueError: pass
    return {
        "gpu_memory_peak_mib_by_physical_gpu": {str(g): max((x[0] for x in gpu), default=None) for g in (0,1)},
        "gpu_utilization_peak_percent_by_physical_gpu": {str(g): max((x[1] for x in gpu), default=None) for g in (0,1)},
        "aggregate_process_group_rss_peak_mib": (max(rss)/1024 if rss else None),
        "aggregate_process_group_rss_samples": len(rss),
    }


memory = {"on": memory_stats("on", "03"), "off": memory_stats("off", "01")}
capture_overhead["gpu_memory_peak_mib"] = {
    gpu: memory["on"]["gpu_memory_peak_mib_by_physical_gpu"][gpu] - memory["off"]["gpu_memory_peak_mib_by_physical_gpu"][gpu]
    for gpu in ("0", "1")
}
capture_overhead["aggregate_process_group_rss_peak_mib"] = relative_delta(
    memory["on"]["aggregate_process_group_rss_peak_mib"],
    memory["off"]["aggregate_process_group_rss_peak_mib"],
)
checks["on_formal_success_rate"] = f"{performance.get('on',{}).get('formal_successes',0)}/6"
checks["off_formal_success_rate"] = f"{performance.get('off',{}).get('formal_successes',0)}/6"
result = {
    "status": "PASS" if not errors else "FAIL",
    "errors": errors,
    "checks": checks,
    "parity": parity,
    "performance": performance,
    "capture_overhead_on_minus_off": capture_overhead,
    "memory": memory,
    "prefix_cache_hit_tokens": prefix_cache_hits,
    "interpretation": {
        "algorithm_configuration_changed": False,
        "on_off_difference_is_capture_overhead": True,
        "api_internal_scheduler_request_id": "UNKNOWN; public response id gives exact one-request association",
        "cpu_memory_measurement": "sum of server-process-group RSS; shared pages may be counted more than once",
        "prompt_decode_mapping": "prefill rows map frozen prompt IDs; decode rows map processed generated input IDs and predict the following output token",
        "last_output_token_route": "absent if not processed by a subsequent model forward",
        "tp1_route_comparison": f"diagnostic only: {tp_mismatches} of {tp_compared} prompt token/layer route rows differ between TP=1 and TP=2 despite exact token-ID alignment; this does not change the ON/OFF TP=2 parity or single-logical-array dedup checks",
    },
}
(ROOT / "independent_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with (ROOT / "quality_trace.jsonl").open("a", encoding="utf-8") as _:
    pass
print(json.dumps(result, indent=2, ensure_ascii=False))
if errors:
    raise SystemExit(1)
