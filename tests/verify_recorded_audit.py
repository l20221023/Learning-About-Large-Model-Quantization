"""Cross-check the recorded official audit offline against its raw inputs."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from math import prod
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    require(hashlib.sha256(index_bytes).hexdigest() == manifest["index"]["sha256"],
            "index digest mismatch")
    config_bytes = (root / "config.json").read_bytes()
    require(hashlib.sha256(config_bytes).hexdigest() == manifest["config"]["sha256"],
            "config digest mismatch")
    require(manifest["index"]["repo_commit"] == manifest["config"]["repo_commit"]
            == "85864749cd611b4353ce1decdb286193298f64c7" == manifest["requested_revision"],
            "index or config revision mismatch")
    index = json.loads(index_bytes)
    config = json.loads(config_bytes)
    topology = {
        "num_hidden_layers": 27, "first_k_dense_replace": 1, "moe_layer_freq": 1,
        "n_routed_experts": 64, "n_shared_experts": 2, "num_experts_per_tok": 6,
        "hidden_size": 2048, "intermediate_size": 10944, "moe_intermediate_size": 1408,
        "torch_dtype": "bfloat16",
    }
    require(all(config[key] == value for key, value in topology.items()), "config topology mismatch")
    rows = {}
    with (root / "expert_weight_mapping.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["tensor_key"]
            require(key not in rows, f"duplicate CSV key: {key}")
            rows[key] = row
    require(rows.keys() == index["weight_map"].keys(), "CSV and index key sets differ")
    require(manifest["status"] == "PASS" and len(rows) == 5291 and len(manifest["shards"]) == 4,
            "unexpected audit status or inventory")
    with (root / "mapping_differences.csv").open(encoding="utf-8", newline="") as stream:
        require(not list(csv.DictReader(stream)), "unexpected nonempty differences")
    seen = set()
    payload_bytes = 0
    transferred = 0
    for name, record in manifest["shards"].items():
        raw = (root / record["header"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == record["header_sha256"], f"{name}: hash mismatch")
        require(len(raw) == record["header_bytes"], f"{name}: length mismatch")
        header = json.loads(raw)
        header.pop("__metadata__", None)
        cursor = 0
        for key, tensor in sorted(header.items(), key=lambda item: item[1]["data_offsets"]):
            require(key not in seen and index["weight_map"][key] == name, f"{key}: wrong shard")
            seen.add(key)
            row = rows[key]
            start, end = tensor["data_offsets"]
            require(start == cursor and end - start == 2 * prod(tensor["shape"]),
                    f"{key}: offset span mismatch")
            cursor = end
            require(
                row["status"] == "VERIFIED" and row["evidence"] == "[D][I][H]"
                and row["shard"] == row["header_shard"] == name
                and row["dtype"] == row["expected_dtype"] == tensor["dtype"] == "BF16"
                and json.loads(row["actual_shape"]) == json.loads(row["expected_shape"]) == tensor["shape"]
                and (int(row["offset_start"]), int(row["offset_end"])) == (start, end),
                f"{key}: CSV and Header mismatch",
            )
        require(cursor + 8 + len(raw) == record["file_bytes"], f"{name}: file coverage mismatch")
        first, second = record["requests"]
        require(first["http_status"] == second["http_status"] == 206
                and first["repo_commit"] == second["repo_commit"] == manifest["requested_revision"]
                and first["response_bytes"] == 8 and second["response_bytes"] == len(raw)
                and first["content_range"] == f"bytes 0-7/{record['file_bytes']}"
                and second["content_range"] == f"bytes 8-{7 + len(raw)}/{record['file_bytes']}"
                and first["etag"] == second["etag"] and first["linked_etag"] == second["linked_etag"],
                f"{name}: Range evidence mismatch")
        payload_bytes += cursor
        transferred += 8 + len(raw)
    require(seen == rows.keys(), "Header inventory does not cover the CSV")
    require(payload_bytes == index["metadata"]["total_size"] == 31_412_968_448,
            "total payload size mismatch")
    roles = Counter(row["role"] for row in rows.values())
    routed = Counter((int(r["layer_id"]), int(r["expert_id"])) for r in rows.values()
                     if r["role"] == "routed")
    require(routed == {(layer, expert): 3 for layer in range(1, 27) for expert in range(64)},
            "routed layer/expert coverage mismatch")
    print(json.dumps({"tensors": len(seen), "roles": roles, "routed_experts": len(routed),
                      "payload_bytes_from_offsets": payload_bytes, "header_range_bytes": transferred,
                      "total_response_body_bytes": transferred + len(index_bytes) + len(config_bytes)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    verify(parser.parse_args().audit_dir)
