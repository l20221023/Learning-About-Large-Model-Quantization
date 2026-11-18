#!/usr/bin/env python3
"""Validate raw records and summarize only directly observable metrics."""
import argparse
import json
import statistics
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(rows):
    assert len(rows) == 15, f"expected 15 requests, got {len(rows)}"
    assert sum(r["phase"] == "quality" for r in rows) == 6
    assert sum(r["phase"] == "perf" for r in rows) == 9
    assert len({(r["phase"], r["group"], r["prompt_index"], r["repeat"]) for r in rows}) == 15
    ok = [r for r in rows if r["status"] == "ok"]
    perf = [r for r in ok if r["phase"] == "perf"]
    for row in ok:
        assert row["e2e_s"] >= 0
        assert row["chunk_count"] == len(row["chunks"])
        assert row["output_text"] == "".join(c["text"] for c in row["chunks"])
        if row["ttft_s"] is not None:
            assert 0 <= row["ttft_s"] <= row["e2e_s"]
    token_rates = []
    for row in perf:
        count = (row.get("usage") or {}).get("completion_tokens")
        if isinstance(count, int) and count > 0 and row["e2e_s"] > 0:
            token_rates.append(count / row["e2e_s"])
    return {"requests": len(rows), "successful": len(ok), "failed": len(rows) - len(ok),
            "perf_successful": len(perf),
            "ttft_s_p50": percentile([r["ttft_s"] for r in perf if r["ttft_s"] is not None], .5),
            "ttft_s_p95": percentile([r["ttft_s"] for r in perf if r["ttft_s"] is not None], .95),
            "e2e_s_p50": percentile([r["e2e_s"] for r in perf], .5),
            "e2e_s_p95": percentile([r["e2e_s"] for r in perf], .95),
            "tokens_per_e2e_s_median": statistics.median(token_rates) if token_rates else None,
            "token_rate_sample_count": len(token_rates),
            "itl_tpot": None, "itl_reason": "SSE chunks lack verified one-token-per-event mapping"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    result = summarize(rows)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
