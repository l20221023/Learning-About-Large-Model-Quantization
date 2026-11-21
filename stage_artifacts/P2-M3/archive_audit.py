#!/usr/bin/env python3
"""Verify preserved M3 snapshot, source copy, O2 references, and control evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M3 = ROOT / "stage_artifacts/P2-M3"
O2 = ROOT / "stage_artifacts/OffloadGate/BF16_decode_20261120T172330"
O2_TRACE = O2 / "remote-evidence/OffloadGate-BF16-decode-20261120T172330/trace"
errors = []


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


original_manifest = M3 / "SHA256SUMS.txt"
verified_original = []
for line_no, line in enumerate(original_manifest.read_text(encoding="utf-8").splitlines(), 1):
    parts = line.split("  ", 1)
    if len(parts) != 2 or len(parts[0]) != 64:
        errors.append(f"original manifest malformed at line {line_no}")
        continue
    rel = parts[1].replace("/", "\\")
    path = M3 / rel
    if not path.is_file():
        errors.append(f"original manifest file missing: {parts[1]}")
    elif sha(path) != parts[0]:
        errors.append(f"original manifest hash mismatch: {parts[1]}")
    else:
        verified_original.append(parts[1])

source_lines = (M3 / "vllm_source_sha256.txt").read_text(encoding="utf-8").splitlines()
source_verified = []
prefix = "/site-packages/vllm/"
for line in source_lines:
    fields = line.split("  ", 1)
    if len(fields) != 2 or prefix not in fields[1]:
        errors.append(f"source hash record malformed: {line}")
        continue
    rel = fields[1].split(prefix, 1)[1]
    local = M3 / "source_snapshot/vllm" / rel
    if not local.is_file() or sha(local) != fields[0]:
        errors.append(f"source snapshot hash mismatch or missing: {rel}")
    else:
        source_verified.append(rel)

m3_audit = json.loads((M3 / "independent_audit.json").read_text(encoding="utf-8"))
tp_audit = json.loads((M3 / "tp_control/tp_control_audit.json").read_text(encoding="utf-8"))
analysis = json.loads((M3 / "tp_routing_difference_analysis.json").read_text(encoding="utf-8"))
diff_report = (M3 / "tp_routing_difference_analysis.md").read_text(encoding="utf-8")
closeout_report = (M3 / "m3_closeout.md").read_text(encoding="utf-8")
o2_audit = json.loads((O2 / "independent_audit.json").read_text(encoding="utf-8"))
o2_replay = json.loads((ROOT / "stage_artifacts/O2_acceptance_20261120/replay_v1/audit.json").read_text(encoding="utf-8"))
o2_sim = json.loads((O2 / "cache_simulation.json").read_text(encoding="utf-8"))
if m3_audit.get("status") != "PASS" or m3_audit.get("errors"):
    errors.append("original M3 independent audit is not clean PASS")
if tp_audit.get("status") != "PASS" or tp_audit.get("errors"):
    errors.append("controlled TP audit is not clean PASS")
if analysis.get("analysis_status") != "PASS" or analysis.get("causal_attribution_status") != "PARTIAL":
    errors.append("TP analysis status does not preserve its partial causal limitation")
checks = analysis.get("alignment", {})
if (checks.get("prefill_rows"), checks.get("prefill_set_mismatches"), analysis.get("by_phase", {}).get("prefill", {}).get("ordered_mismatches")) != (4550, 388, 688):
    errors.append("historical prefill set/order counts differ from recorded result")
if (checks.get("decode_rows"), checks.get("decode_set_mismatches"), analysis.get("by_phase", {}).get("decode", {}).get("ordered_mismatches")) != (2054, 178, 332):
    errors.append("history-aligned decode set/order counts differ from recorded result")
if "48/702" not in diff_report or "93/702" not in diff_report:
    errors.append("controlled experiment counts missing from the analysis report")
if "PARTIAL" not in closeout_report or "因果归因" not in closeout_report:
    errors.append("closeout report does not mark incomplete causal attribution")
for layer in range(1, 27):
    p = analysis["by_phase_layer"][f"prefill:layer-{layer}"]
    d = analysis["by_phase_layer"][f"decode:layer-{layer}"]
    expected_line = f"| {layer} | {p['set_mismatches']}/175 | {p['ordered_mismatches']}/175 | {d['set_mismatches']}/79 | {d['ordered_mismatches']}/79 |"
    if expected_line not in diff_report:
        errors.append(f"analysis report layer table differs from recalculated rows at layer {layer}")
for policy in ("lru", "lfu"):
    k6 = next(x for x in o2_sim["results"] if x["policy"] == policy and x["capacity_experts_per_layer"] == 6)
    k64 = next(x for x in o2_sim["results"] if x["policy"] == policy and x["capacity_experts_per_layer"] == 64)
    if (k6["hits"], k6["misses"]) != (7405, 21611) or (k64["hits"], k64["misses"]) != (21136, 7880):
        errors.append(f"O2 {policy.upper()} reuse figures differ from raw simulation")
if o2_audit.get("status") != "PASS":
    errors.append("O2 source audit is not PASS")
if o2_replay.get("status") != "PASS" or len(o2_replay.get("cells", [])) != 18:
    errors.append("O2 independent cache replay is not 18-cell PASS")

o2_decode = O2_TRACE / "decode_trace.jsonl"
o2_trace_hash = sha(o2_decode) if o2_decode.is_file() else None
expected_o2_trace_hash = "dbe9920be8ecbfa8b6828e12d8c8921a53933c001b5ae02b001693b6f7a66993"
if o2_trace_hash != expected_o2_trace_hash:
    errors.append("O2 decode trace SHA256 differs from its recorded fingerprint")

remote_file_counts = {}
for tp in (1, 2):
    name = f"tp-control-tp{tp}-20261120"
    folder = M3 / "tp_control/remote" / name
    sidecar = M3 / "tp_control/remote" / f"{name}.sha256"
    count = 0
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not parts[1].startswith("./"):
            errors.append(f"TP{tp}: malformed remote sidecar entry")
            continue
        target = folder / parts[1][2:].replace("/", "\\")
        if not target.is_file() or sha(target) != parts[0]:
            errors.append(f"TP{tp}: remote/local evidence hash mismatch: {parts[1]}")
        else:
            count += 1
    remote_file_counts[str(tp)] = count
    if count != 20:
        errors.append(f"TP{tp}: expected 20 hash-verified copied run files, got {count}")

result = {
    "status": "PASS" if not errors else "FAIL",
    "errors": errors,
    "original_m3_manifest": {
        "entries": len(original_manifest.read_text(encoding="utf-8").splitlines()),
        "verified_entries": len(verified_original),
        "manifest_sha256": sha(original_manifest),
    },
    "vllm_source_snapshot": {"expected_records": len(source_lines), "verified_files": len(source_verified), "paths": source_verified},
    "original_m3_audit_status": m3_audit.get("status"),
    "tp_control_audit_status": tp_audit.get("status"),
    "tp_routing_causal_status": analysis.get("causal_attribution_status"),
    "o2_trace_sha256": o2_trace_hash,
    "o2_trace_audit_status": o2_audit.get("status"),
    "o2_cache_replay_cells": len(o2_replay.get("cells", [])),
    "control_run_files_remote_local_hash_verified": remote_file_counts,
}
(M3 / "archive_freeze_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    raise SystemExit(1)
