"""Load a tiny random checkpoint through the pinned native HF implementation."""

import argparse
import importlib.metadata
import sys
from pathlib import Path
from shutil import copyfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from p2.trace.router import trace_prefill, write_trace
from p2.trace.runtime import (
    SOURCE_SHA256, TOKENIZER_SHA256, digest, encode_chat_prompt, get_native_gates, load_local_model,
)
from p2.weights.audit import write_json
from p2.weights.checkpoint import MODEL, REVISION
from p2.weights.safetensors_io import now

SEED = 20261117
LAYERS = 4


def run(source, scratch, out):
    out.mkdir(parents=True, exist_ok=False)
    start = now()
    for name, expected in (SOURCE_SHA256 | TOKENIZER_SHA256).items():
        if digest(source / name) != expected:
            raise ValueError(f"{name}: source differs from pinned revision")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    config = AutoConfig.from_pretrained(str(source.resolve()), trust_remote_code=True,
                                        local_files_only=True)
    config.update({
        "vocab_size": 32, "hidden_size": 32, "intermediate_size": 40,
        "moe_intermediate_size": 12, "num_hidden_layers": LAYERS,
        "num_attention_heads": 4, "num_key_value_heads": 4, "kv_lora_rank": 8,
        "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
        "max_position_embeddings": 64, "rope_scaling": None,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    })
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="eager")
    scratch.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(scratch, safe_serialization=True, max_shard_size="200KB")
    for name in ("configuration_deepseek.py", "modeling_deepseek.py"):
        copyfile(source / name, scratch / name)
    shards = sorted(scratch.glob("model-*-of-*.safetensors"))
    if len(shards) < 2:
        raise ValueError("expected a sharded synthetic checkpoint")
    del model
    model = load_local_model(scratch, "cpu")
    gates = get_native_gates(model, expected_layers=LAYERS)
    inputs = {
        "input_ids": torch.tensor([[1, 4, 5, 2, 0], [1, 7, 8, 9, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
    }
    special = [0, 1, 2]
    rows, metrics = trace_prefill(model, gates, inputs, excluded_token_ids=special)
    if metrics["valid_tokens"] != 5 or metrics["records"] != 15 or gates[1].weight.dtype != torch.bfloat16:
        raise ValueError("unexpected tiny checkpoint dtype, token count or trace size")
    prompt = "Explain what an expert router does in a mixture-of-experts model in one sentence."
    encoded, official_special, tokenizer_details = encode_chat_prompt(source, prompt, max_tokens=64)
    prompt_ids = encoded["input_ids"].tolist()[0]
    if prompt_ids[0] != 100000 or any(x >= 102400 for x in prompt_ids):
        raise ValueError("unexpected official tokenizer output for fixed probe prompt")
    tokenizer_details.update(
        input_ids=prompt_ids, special_token_ids=official_special, metadata_sha256=TOKENIZER_SHA256,
        input_tokens=len(prompt_ids), non_special_tokens=sum(i not in official_special for i in prompt_ids),
    )
    write_trace(out / "router_trace.jsonl", rows, evidence="SYNTHETIC")
    manifest = {
        "status": "PASS", "run_id": out.name, "started_at": start, "finished_at": now(),
        "evidence": "SYNTHETIC", "native_code_from": MODEL, "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256, "weights": "random initialization; no pretrained data",
        "synthetic_checkpoint_sha256": {path.name: digest(path) for path in shards},
        "synthetic_index_sha256": digest(scratch / "model.safetensors.index.json"),
        "phase": "prefill", "use_cache": False, "backend": "HF Remote Code + PyTorch eager",
        "device": "cpu", "seed": SEED, "num_cpu_threads": 1, "trace": "router_trace.jsonl",
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ("torch", "transformers", "accelerate", "safetensors")},
        "inputs": {key: value.tolist() for key, value in inputs.items()},
        "excluded_token_ids": special, "metrics": metrics,
        "actual_model_class": type(model).__name__,
        "actual_attention_class": type(model.model.layers[0].self_attn).__name__,
        "actual_moe_class": type(model.model.layers[1].mlp).__name__,
        "actual_gate_class": type(gates[1]).__name__, "parameter_dtype": str(gates[1].weight.dtype),
        "router_logits_source": "float32 linear replay of native gate input and weight",
        "topk_source": "native gate forward output",
        "official_tokenizer_probe": tokenizer_details,
    }
    write_json(out / "manifest.json", manifest)
    print(f"SYNTHETIC {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.source_dir, args.scratch_dir, args.output_dir)
