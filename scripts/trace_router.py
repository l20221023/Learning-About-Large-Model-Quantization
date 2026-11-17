"""Run a bounded prefill Router Trace with an existing official BF16 snapshot."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2.weights.audit import write_json
from p2.weights.checkpoint import MODEL, REVISION
from p2.weights.safetensors_io import now

PROMPT = "Explain what an expert router does in a mixture-of-experts model in one sentence."
SEED = 20261117


def run(args):
    import torch
    from p2.trace.router import trace_prefill, write_trace
    from p2.trace.runtime import (
        encode_chat_prompt, environment, get_native_gates, load_local_model, preflight,
    )

    if args.mode == "probe":
        print(json.dumps(environment(), indent=2, allow_nan=False))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": now(), "run_id": args.output_dir.name,
        "model": MODEL, "revision": REVISION, "planned_weight_dtype": "BF16",
        "backend": "HF Remote Code + PyTorch eager", "device": args.device,
        "snapshot": str(args.snapshot.resolve()), "phase": "prefill", "use_cache": False,
        "prompt": PROMPT, "seed": SEED, "max_input_tokens": 64,
        "evidence": "UNVERIFIED",
    }
    status, exit_code = "FAIL", 1
    try:
        current_runtime = environment()
        manifest["environment"] = current_runtime
        problems, snapshot_evidence = preflight(args.snapshot, args.device, current_runtime)
        manifest.update(preflight_issues=problems, snapshot_evidence=snapshot_evidence)
        if problems:
            status, exit_code = "BLOCKED", 2
        elif args.dry_run:
            status, exit_code = "PREFLIGHT_OK", 0
        else:
            torch.manual_seed(SEED)
            torch.set_float32_matmul_precision("highest")
            encoded, special, tokenizer_details = encode_chat_prompt(
                args.snapshot, PROMPT, max_tokens=manifest["max_input_tokens"]
            )
            model = load_local_model(args.snapshot, args.device)
            gates = get_native_gates(model, expected_layers=27)
            input_device = model.get_input_embeddings().weight.device
            inputs = {key: encoded[key].to(input_device) for key in ("input_ids", "attention_mask")}
            rows, metrics = trace_prefill(model, gates, inputs, excluded_token_ids=special)
            evidence = "MEASURED_PRETRAINED_BF16"
            write_trace(args.output_dir / "router_trace.jsonl", rows, evidence=evidence)
            manifest.update(
                evidence=evidence, trace="router_trace.jsonl",
                metrics=metrics, actual_attention_class=type(model.model.layers[0].self_attn).__name__,
                actual_moe_class=type(model.model.layers[1].mlp).__name__,
                parameter_dtype=str(gates[1].weight.dtype), excluded_token_ids=special,
                inputs={name: tensor.tolist() for name, tensor in encoded.items()
                        if name in ("input_ids", "attention_mask")},
                router_logits_source="float32 linear replay of native gate input and weight",
                topk_source="native gate forward output",
                tokenizer=tokenizer_details,
            )
            status, exit_code = "PASS", 0
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    manifest.update(status=status, exit_code=exit_code, finished_at=now())
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in
                      ("status", "preflight_issues", "error", "metrics") if key in manifest}))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("probe", help="record CPU, CUDA, display drivers and package availability")
    real = modes.add_parser("real", help="run the pinned model on an explicitly local snapshot")
    real.add_argument("--snapshot", required=True, type=Path)
    real.add_argument("--device", required=True, choices=("cpu", "cuda"))
    real.add_argument("--output-dir", required=True, type=Path, help="new run directory")
    real.add_argument("--dry-run", action="store_true", help="check local files and hardware only")
    return run(parser.parse_args())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ImportError, OSError, ValueError) as error:
        sys.exit(str(error))
