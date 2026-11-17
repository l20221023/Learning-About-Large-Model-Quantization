"""Fetch pinned DeepSeek Python code and small metadata for the CPU source probe."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2.weights.checkpoint import BASE_URL, REVISION
from p2.weights.safetensors_io import request_bytes

FILES = (
    "LICENSE", "config.json", "configuration_deepseek.py", "modeling_deepseek.py",
    "tokenization_deepseek_fast.py", "tokenizer_config.json", "tokenizer.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for name in FILES:
        limits = {"max_metadata_bytes": 6 * 1024 * 1024} if name == "tokenizer.json" else {}
        raw, record = request_bytes(f"{BASE_URL}/{name}", **limits)
        if record["repo_commit"] != REVISION:
            raise ValueError(f"{name}: missing pinned revision attestation")
        (args.output_dir / name).write_bytes(raw)
        records[name] = dict(record, sha256=hashlib.sha256(raw).hexdigest())
    print(json.dumps({"revision": REVISION, "files": records}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
