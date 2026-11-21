#!/usr/bin/env python3
"""Independent sample/data integrity audit against frozen Parquet sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference(task: str, row: dict):
    if task == "gsm8k":
        m = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", row["answer"])
        return m.group(1).replace(",", "") if m else None
    answer = row["answer"]
    if isinstance(answer, int):
        return "ABCD"[answer] if 0 <= answer < 4 else None
    m = re.fullmatch(r"(?:答案[:：]?\s*)?([ABCD])", str(answer).strip().upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("quality_config.yaml"))
    ap.add_argument("--manifest", type=Path, default=Path("dataset_manifest.json"))
    ap.add_argument("--samples", type=Path, default=Path("quality_samples.jsonl"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    manifest_map = {}
    row_cache = {}
    revisions = {"gsm8k": config["datasets"]["gsm8k"]["revision"],
                 "ceval": config["datasets"]["ceval"]["revision"],
                 "mmlu": config["datasets"]["mmlu"]["revision"]}
    for task, item in manifest["datasets"].items():
        if item["revision"] != revisions[task]:
            errors.append(f"{task}: manifest/config revision mismatch")
        for file in item["files"]:
            path = args.data_dir / file["local_path"]
            if not path.is_file():
                errors.append(f"missing source file {path}")
                continue
            actual = sha256(path)
            if actual != file["sha256"]:
                errors.append(f"source SHA256 mismatch {path}")
            if path.stat().st_size != file["size_bytes"]:
                errors.append(f"source byte-size mismatch {path}")
            key = (task, path.as_posix())
            manifest_map[key] = file["sha256"]
            table = pq.read_table(path).to_pylist()
            row_cache[key] = table
            if len(table) != file["rows"]:
                errors.append(f"source row count mismatch {path}")
            if task in ("gsm8k", "mmlu") and file["sha256"] != config["datasets"][task]["file_sha256"]:
                errors.append(f"{task}: file SHA differs from pinned config")
    if len(manifest["datasets"].get("ceval", {}).get("files", [])) != config["datasets"]["ceval"]["source_file_count"]:
        errors.append("C-Eval source shard count differs from pinned config")
    seen = set()
    counts = Counter()
    by_subject = defaultdict(Counter)
    for sample in samples:
        task = sample["dataset"]
        key_id = (task, sample["sample_id"])
        if key_id in seen:
            errors.append(f"duplicate sample ID {key_id}")
        seen.add(key_id)
        counts[task] += 1
        by_subject[task][sample["subject"]] += 1
        if sample["seed"] != 42 or sample["dataset_revision"] != revisions[task]:
            errors.append(f"seed/revision mismatch {key_id}")
        source_key = (task, (args.data_dir / sample["source_file"]).as_posix())
        expected_file_sha = manifest_map.get(source_key)
        if expected_file_sha is None or expected_file_sha != sample["source_file_sha256"]:
            errors.append(f"source file SHA mismatch {key_id}")
            continue
        source_rows = row_cache[source_key]
        index = sample["source_row_index"]
        if not isinstance(index, int) or not 0 <= index < len(source_rows):
            errors.append(f"bad source row index {key_id}")
            continue
        row = source_rows[index]
        if sample["reference_answer"] != reference(task, row):
            errors.append(f"reference answer mismatch {key_id}")
        if sample["subject"] != (str(row.get("subject")) if task == "mmlu" else sample["subject"]):
            errors.append(f"subject mismatch {key_id}")
        template = config["datasets"][task]["prompt_template"]
        prompt_fields = {"question": row["question"]}
        if task == "ceval":
            prompt_fields.update({k: row[k] for k in "ABCD"})
            expected_choices = [row[k] for k in "ABCD"]
        elif task == "mmlu":
            prompt_fields.update(dict(zip("ABCD", row["choices"], strict=True)))
            expected_choices = list(row["choices"])
        else:
            expected_choices = None
        if sample["prompt"] != template.format(**prompt_fields):
            errors.append(f"prompt mismatch {key_id}")
        if sample.get("choices") != expected_choices:
            errors.append(f"choice list mismatch {key_id}")
        if sample.get("source_dataset_id") != row.get("id"):
            errors.append(f"original dataset ID mismatch {key_id}")
    for task in ("gsm8k", "ceval", "mmlu"):
        if counts[task] != 256:
            errors.append(f"{task} sample count {counts[task]} != 256")
        if task != "gsm8k" and by_subject[task]:
            allocations = list(by_subject[task].values())
            if max(allocations) - min(allocations) > 1:
                errors.append(f"{task} stratification imbalance exceeds 1: {dict(by_subject[task])}")
    if len(samples) != 768:
        errors.append(f"total sample count {len(samples)} != 768")
    print("counts", dict(counts))
    print("subject_counts", {task: dict(counts_) for task, counts_ in by_subject.items()})
    print("sample_sha256", sha256(args.samples))
    print("dataset_manifest_sha256", sha256(args.manifest))
    if errors:
        print("FAIL", len(errors))
        for error in errors[:100]:
            print(error)
        raise SystemExit(1)
    print("PASS: source hashes, row IDs, labels, prompts, choices, revisions, stratification, and 768 count")


if __name__ == "__main__":
    main()
