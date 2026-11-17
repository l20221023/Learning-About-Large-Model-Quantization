"""Write a checkpoint audit, including partial evidence on read failures."""

import csv
import hashlib
import json
from collections import Counter
from math import prod

from p2.weights.checkpoint import (
    BASE_URL, INDEX_NAME, MODEL, REVISION, SHARDS, TOTAL_BYTES,
    expected_weights, json_object, local_shard, parse_index,
)
from p2.weights.safetensors_io import (
    DTYPE_BYTES, MAX_METADATA_BYTES, now, read_local_header, read_remote_header, request_bytes,
)

FIELDS = (
    "layer_id", "expert_id", "role", "tensor_key", "shard", "header_shard",
    "expected_shape", "actual_shape", "expected_dtype", "dtype",
    "offset_start", "offset_end", "evidence", "status", "issue",
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare(expected, weights, headers):
    locations = {}
    for shard, header in headers.items():
        for key in header.tensors:
            locations.setdefault(key, []).append(shard)
    for key in sorted(expected.keys() | weights.keys() | locations.keys()):
        prediction = expected.get(key)
        shard = weights.get(key)
        found = locations.get(key, [])
        observed = shard if shard in found else next(iter(found), None)
        actual = headers[observed].tensors[key] if observed else None
        issues = []
        if prediction is None:
            issues.append("unexpected_tensor")
        if weights and shard is None:
            issues.append("missing_from_index")
        if shard is not None and shard not in SHARDS:
            issues.append("unexpected_shard")
        if shard in headers and shard not in found:
            issues.append("missing_from_header")
        if len(found) > 1:
            issues.append("duplicate_in_headers")
        if shard and any(shard != name for name in found):
            issues.append("shard_mismatch")
        if actual and prediction:
            if actual.shape != prediction.shape:
                issues.append("shape_mismatch")
            if actual.dtype != prediction.dtype:
                issues.append("dtype_mismatch")
        status = "UNKNOWN"
        if issues:
            status = "MISMATCH"
        elif actual and shard:
            status = "VERIFIED"
        elif shard:
            status = "PREDICTED"
        evidence = [tag for tag, present in (("[D]", prediction), ("[I]", shard), ("[H]", actual))
                    if present]
        if status in ("PREDICTED", "UNKNOWN"):
            evidence.append("[U]")
        yield {
            "layer_id": prediction.layer_id if prediction else None,
            "expert_id": prediction.expert_id if prediction else None,
            "role": prediction.role if prediction else None, "tensor_key": key,
            "shard": shard, "header_shard": "|".join(found),
            "expected_shape": json.dumps(prediction.shape) if prediction else None,
            "actual_shape": json.dumps(actual.shape) if actual else None,
            "expected_dtype": prediction.dtype if prediction else None,
            "dtype": actual.dtype if actual else None,
            "offset_start": actual.offsets[0] if actual else None,
            "offset_end": actual.offsets[1] if actual else None,
            "evidence": "".join(evidence), "status": status, "issue": "|".join(issues),
        }


def audit(out, *, index_path, shard_dir, remote):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"model": MODEL, "requested_revision": REVISION, "started_at": now(),
                "index": {}, "config": {}, "shards": {}, "issues": []}
    expected = expected_weights()
    weights = {}
    index_size = None
    try:
        if index_path:
            with index_path.open("rb") as stream:
                raw = stream.read(MAX_METADATA_BYTES + 1)
            source = {"path": str(index_path.resolve()), "retrieved_at": now(),
                      "revision_attestation": "local input supplied by operator"}
        else:
            raw, source = request_bytes(f"{BASE_URL}/{INDEX_NAME}")
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("index exceeds metadata byte limit")
        index = parse_index(raw)
        weights = index["weight_map"]
        index_size = index.get("metadata", {}).get("total_size") if isinstance(index.get("metadata"), dict) else None
        (out / INDEX_NAME).write_bytes(raw)
        manifest["index"] = dict(source, sha256=hashlib.sha256(raw).hexdigest(),
                                 tensors=len(weights), payload_bytes=index_size,
                                 shard_counts=dict(Counter(weights.values())))
        if set(weights.values()) != set(SHARDS) or index_size != TOTAL_BYTES:
            manifest["issues"].append("index shard set or total_size differs from pinned checkpoint")
    except (OSError, ValueError) as error:
        manifest["index"]["error"] = str(error)
    if remote or index_path is None:
        try:
            raw, source = request_bytes(f"{BASE_URL}/config.json")
            config = json_object(raw)
            (out / "config.json").write_bytes(raw)
            manifest["config"] = dict(source, sha256=hashlib.sha256(raw).hexdigest(),
                                      torch_dtype=config.get("torch_dtype"))
            if config.get("torch_dtype") != "bfloat16":
                manifest["issues"].append("official config torch_dtype differs from BF16 expectation")
        except (OSError, ValueError) as error:
            manifest["config"]["error"] = str(error)
    headers = {}
    if shard_dir or remote:
        (out / "headers").mkdir()
    for name in SHARDS:
        record = {"requests": []}
        manifest["shards"][name] = record
        try:
            if shard_dir:
                path = local_shard(shard_dir, name)
                record.update(source="local", path=str(path), retrieved_at=now(),
                              revision_attestation="local input supplied by operator")
                with path.open("rb") as stream:
                    header, raw = read_local_header(stream)
            elif remote:
                record.update(source="http_range", url=f"{BASE_URL}/{name}")
                header, raw = read_remote_header(record["url"], record["requests"])
            else:
                record["error"] = "no Header source selected"
                continue
            basename = name.removesuffix(".safetensors") + ".header.json"
            (out / "headers" / basename).write_bytes(raw)
            headers[name] = header
            record.update(
                header=f"headers/{basename}", header_bytes=header.length,
                header_sha256=header.sha256, file_bytes=header.file_size,
                payload_bytes=header.file_size - 8 - header.length, tensors=len(header.tensors),
            )
        except (OSError, ValueError) as error:
            record["error"] = str(error)
    rows = list(compare(expected, weights, headers))
    counts = Counter(row["status"] for row in rows)
    for filename, selected in (
        ("expert_weight_mapping.csv", rows),
        ("mapping_differences.csv", (row for row in rows if row["status"] != "VERIFIED")),
    ):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(selected)
    actual_payload = sum(h.file_size - 8 - h.length for h in headers.values())
    expected_payload = sum(prod(t.shape) * DTYPE_BYTES[t.dtype] for t in expected.values())
    if len(headers) == len(SHARDS) and actual_payload != index_size and weights:
        manifest["issues"].append("actual payload total differs from index metadata.total_size")
    if counts["MISMATCH"] or manifest["issues"]:
        status, exit_code = "FAIL", 1
    elif len(headers) == len(SHARDS) and counts["VERIFIED"] == len(expected):
        status, exit_code = "PASS", 0
    else:
        status, exit_code = "PARTIAL", 2
    manifest.update(
        status=status, exit_code=exit_code, status_counts=dict(counts),
        role_counts=dict(Counter(row["role"] for row in rows if row["role"])),
        expected_payload_bytes=expected_payload, measured_header_payload_bytes=actual_payload,
        finished_at=now(),
    )
    write_json(out / "manifest.json", manifest)
    return manifest
