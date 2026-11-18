#!/usr/bin/env python3
"""Collect a small, auditable AWQ baseline from a local vLLM OpenAI server."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def stream_chat(url, model, prompt, max_tokens, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    chunks, output, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                part = delta.get("content") or ""
                if part:
                    chunks.append({"offset_s": time.monotonic() - start, "text": part})
                    output.append(part)
                finish = choice.get("finish_reason") or finish
    end = time.monotonic()
    return {"output_text": "".join(output), "chunks": chunks, "usage": usage,
            "finish_reason": finish, "ttft_s": chunks[0]["offset_s"] if chunks else None,
            "e2e_s": end - start, "chunk_count": len(chunks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="p2-awq")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    path = Path(args.workload)
    source = path.read_bytes()
    groups = json.loads(source)["groups"]
    assert set(groups) == {"general", "chinese", "code"}
    assert all(len(items) == 2 and all(isinstance(p, str) and p.strip() for p in items)
               for items in groups.values())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"workload_sha256": hashlib.sha256(source).hexdigest(),
                "workload": str(path), "model": args.model, "url": args.url,
                "design": "quality 6 prompts x 64 tokens; perf 3 prompts x 3 repeats x 64 tokens; sequential",
                "metric_note": "stream chunks are not guaranteed to be individual tokens; ITL/TPOT omitted"}
    (out.parent / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with out.open("w", encoding="utf-8") as file:
        for phase in ("quality", "perf"):
            for group, prompts in groups.items():
                for index, prompt in enumerate(prompts if phase == "quality" else prompts[:1]):
                    for repeat in range(1 if phase == "quality" else 3):
                        row = {"phase": phase, "group": group, "prompt_index": index,
                               "repeat": repeat, "prompt": prompt, "max_tokens": 64,
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                        try:
                            row.update(stream_chat(args.url, args.model, prompt, 64, args.timeout))
                            row["status"] = "ok"
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file.flush()
    print(out)


if __name__ == "__main__":
    main()
