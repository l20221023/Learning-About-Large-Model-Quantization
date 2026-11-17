"""Audit the pinned DeepSeek checkpoint without reading any tensor payload."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2.weights.audit import audit
from p2.weights.checkpoint import BASE_URL, INDEX_NAME, SHARDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, help="existing index JSON (default: pinned remote index)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--shard-dir", type=Path, help="directory holding local shard files")
    source.add_argument("--remote-headers", action="store_true", help="fetch bounded HTTP Ranges")
    parser.add_argument("--output-dir", type=Path, help="new directory for this run")
    parser.add_argument("--dry-run", action="store_true", help="print the plan with no reads or writes")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({
            "index": str(args.index) if args.index else f"{BASE_URL}/{INDEX_NAME}",
            "header_source": str(args.shard_dir) if args.shard_dir else (
                BASE_URL if args.remote_headers else None),
            "shards": SHARDS, "output": str(args.output_dir) if args.output_dir else None,
        }, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")
    try:
        result = audit(args.output_dir, index_path=args.index,
                       shard_dir=args.shard_dir, remote=args.remote_headers)
    except (OSError, ValueError) as error:
        print(f"audit error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "status_counts", "issues")}))
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
