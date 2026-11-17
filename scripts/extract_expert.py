"""Read one routed expert from already-present local Safetensors shards."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2.weights.checkpoint import parse_index, routed_weights
from p2.weights.extract import expert_plan, extract_expert
from p2.weights.safetensors_io import MAX_METADATA_BYTES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="existing index JSON")
    parser.add_argument("--shard-dir", type=Path, help="default: index directory")
    parser.add_argument("--layer", type=int, required=True, help="1..26 (model layer ID)")
    parser.add_argument("--expert", type=int, required=True, help="0..63 (router ID)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="resolve keys from index only")
    action.add_argument("--metadata-only", action="store_true", help="read local headers only")
    action.add_argument("--output", type=Path, help="new .safetensors file")
    args = parser.parse_args()
    try:
        expected = routed_weights(args.layer, args.expert)
        with args.index.open("rb") as stream:
            data = stream.read(MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("index exceeds metadata byte limit")
        plan = expert_plan(expected, parse_index(data)["weight_map"])
        if args.output and args.output.suffix != ".safetensors":
            raise ValueError("--output must end in .safetensors")
        rows = plan if args.dry_run else extract_expert(
            plan, args.shard_dir or args.index.parent, args.output
        )
        print(json.dumps({"output": str(args.output) if args.output else None, "tensors": rows},
                         indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"extraction error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
