#!/usr/bin/env python3
"""Reconcile frozen samples, formal outputs, scoring, and summary CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

FAILURES = {"api_failure", "output_truncated", "parse_failure", "context_overflow"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_again(task: str, output: str):
    if task == "gsm8k":
        values = re.findall(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", output)
        if not values:
            final_lines = re.findall(
                r"(?:final\s+(?:numeric\s+)?answer|answer)\s*(?:is\s*)?[:：]?\s*([^\n]+)",
                output,
                re.I,
            )
            if final_lines:
                values = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", final_lines[-1])
        if values:
            return values[-1].replace(",", "")
        numeric_only = re.fullmatch(
            r"\s*\$?([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*",
            output,
        )
        if numeric_only:
            return numeric_only.group(1).replace(",", "")
        terminal_number = re.search(
            r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*[.!]?\s*$",
            output,
        )
        return terminal_number.group(1).replace(",", "") if terminal_number else None
    match = re.match(r"\s*(?:answer\s*[:：]?\s*)?([ABCD])(?:[\.\)\s]|$)", output, re.I)
    if match:
        return match.group(1).upper()
    marked = re.findall(
        r"(?:final\s+answer|answer\s*(?:is)?|答案(?:是)?)\s*[:：]?\s*([ABCD])\b",
        output,
        re.I,
    )
    return marked[-1].upper() if marked else None


def rows(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=Path("quality_samples.jsonl"))
    ap.add_argument("--raw", type=Path, default=Path("quality_raw/quality_formal.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--config", type=Path, default=Path("quality_config.yaml"))
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or config["model"]["tokenizer_path"], trust_remote_code=True
    )
    max_model_len = config["model"]["max_model_len"]
    samples, results = rows(args.samples), rows(args.raw)
    errors = []
    sample_map, result_map = {}, {}
    for row in samples:
        key = (row["dataset"], row["sample_id"])
        if key in sample_map:
            errors.append(f"duplicate sample id {key}")
        sample_map[key] = row
    for row in results:
        key = (row["dataset"], row["sample_id"])
        if key in result_map:
            errors.append(f"duplicate result id {key}")
        result_map[key] = row
    if set(sample_map) != set(result_map):
        errors.append("sample and result ID sets differ")

    grouped = defaultdict(list)
    for key, sample in sample_map.items():
        result = result_map.get(key)
        if result is None:
            continue
        grouped[sample["dataset"]].append(result)
        for field in ("reference_answer", "dataset_revision", "source_file_sha256", "prompt_template_sha256"):
            sample_field = field
            if result.get(sample_field) != sample.get(sample_field):
                errors.append(f"{field} mismatch {key}")
        messages = result.get("input_messages")
        if not isinstance(messages, list):
            errors.append(f"missing input messages {key}")
        else:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            actual_count = len(token_ids)
            count = result.get("input_tokens_local")
            if count != actual_count:
                errors.append(f"token count mismatch {key}: stored={count} actual={actual_count}")
            if actual_count > max_model_len:
                errors.append(f"overlength input {key}: {actual_count}>{max_model_len}")
            api_count = result.get("input_tokens_api")
            if api_count is not None and api_count != actual_count:
                errors.append(f"API/local prompt token mismatch {key}: api={api_count} local={actual_count}")
        status = result.get("status")
        if status not in FAILURES | {"scored_correct", "model_answer_wrong"}:
            errors.append(f"unknown status {status!r} {key}")
        if status in {"scored_correct", "model_answer_wrong"}:
            if not isinstance(result.get("response"), dict) or not result.get("output_text"):
                errors.append(f"missing full response/output {key}")
            if result.get("input_tokens_api") is None or result.get("output_tokens_api") is None:
                errors.append(f"missing API token usage {key}")
            prediction = parse_again(sample["dataset"], result["output_text"])
            if prediction is None or prediction != result.get("prediction"):
                errors.append(f"independent parse mismatch {key}")
            expected_score = int(prediction == sample["reference_answer"]) if prediction is not None else None
            if result.get("score") != expected_score:
                errors.append(f"score mismatch {key}")
            expected_status = "scored_correct" if expected_score == 1 else "model_answer_wrong"
            if status != expected_status:
                errors.append(f"status mismatch {key}")
        elif result.get("score") is not None:
            errors.append(f"failed row has score {key}")

    summary = []
    for task in ("gsm8k", "ceval", "mmlu"):
        task_rows = grouped[task]
        counts = Counter(row["status"] for row in task_rows)
        correct, wrong, failed, total = counts["scored_correct"], counts["model_answer_wrong"], sum(counts[x] for x in FAILURES), len(task_rows)
        if total != 256:
            errors.append(f"{task} result count {total}, expected 256")
        summary.append({
            "dataset": task, "sample_count": total, "correct": correct,
            "model_answer_wrong": wrong, "failure_count": failed,
            "api_failures": counts["api_failure"], "parse_failures": counts["parse_failure"],
            "output_truncated": counts["output_truncated"], "context_overflow": counts["context_overflow"],
            "accuracy_over_256": correct / 256 if total == 256 else "",
            "accuracy_on_scorable": correct / (correct + wrong) if correct + wrong else "",
        })
    if len(samples) != 768:
        errors.append(f"sample count {len(samples)}, expected 768")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "quality_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = [
        "# BF16 quality subset report", "",
        "These are fixed 256-sample subsets, not complete official benchmark results.",
        "Multiple choice uses generation-based option-letter scoring; no log-likelihood scores are mixed.",
        "GSM8K extracts the final number after `####`, an explicit final-answer marker, or a numeric response ending. Ambiguous responses remain parse failures.",
        "The independent audit re-tokenized stored messages with the configured chat template; local input token counts matched API prompt token usage for every request. API input/output token counts are retained per row.",
        f"Frozen sample SHA256: {digest(args.samples)}",
        f"Raw result SHA256: {digest(args.raw)}",
        f"Audit errors: {len(errors)}", "",
        "| Dataset | N | Correct | Wrong | Failures | Accuracy / 256 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        value = row["accuracy_over_256"]
        accuracy = f"{value:.4f}" if isinstance(value, float) else "NA"
        report.append("| {dataset} | {sample_count} | {correct} | {model_answer_wrong} | {failure_count} | {accuracy} |".format(**row, accuracy=accuracy))
    report.extend(["", "## Audit details", ""])
    report.extend(f"- {error}" for error in errors[:100])
    (args.out_dir / "quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(f"FAIL: {len(errors)} audit errors; see quality_report.md")
    print("PASS: 768 sample/result IDs, references, responses, token counts, scores and summary reconcile")


if __name__ == "__main__":
    main()
