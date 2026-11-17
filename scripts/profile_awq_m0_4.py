"""Run curated M0-4 prompt groups through the existing 3090 AWQ snapshot."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

_here = Path(__file__).resolve().parent
_source_root = _here / "src" if (_here / "src").is_dir() else _here.parent / "src"
sys.path.insert(0, str(_source_root))
from p2.weights.audit import write_json
from p2.weights.safetensors_io import now

from trace_awq_3090 import (EVIDENCE, REPO, REV, SEED, install_head_probe,
                            install_router_probe, inventory, pop_head, pop_routes)


def reset_routes(model):
    model._p2_routes = {}
    model._p2_shared = {}


def install_shared_probe(model):
    model._p2_shared = {}
    model._p2_shared_handles = []
    details = {}
    for layer, block in enumerate(model.model.layers[1:], start=1):
        module = getattr(block.mlp, "shared_experts", None)
        if module is None:
            raise ValueError(f"MoE layer {layer} has no shared expert module")
        details[layer] = type(module).__name__

        def capture(_module, inputs, output, *, layer_id=layer):
            if layer_id in model._p2_shared:
                raise ValueError(f"unexpected repeated shared expert call for layer {layer_id}")
            value = inputs[0]
            shape = list(value.shape)
            token_count = math.prod(shape[:-1])
            output_shape = list(output.shape) if hasattr(output, "shape") else None
            model._p2_shared[layer_id] = {
                "module": type(_module).__name__, "calls": 1,
                "input_shape": shape, "output_shape": output_shape,
                "model_input_tokens": token_count,
            }

        model._p2_shared_handles.append(module.register_forward_hook(capture))
    return details


def pop_shared(model):
    result = model._p2_shared
    model._p2_shared = {}
    return result


def remove_shared_probe(model):
    for handle in getattr(model, "_p2_shared_handles", []):
        handle.remove()
    model._p2_shared_handles = []
    model._p2_shared = {}
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    workload_bytes = args.workload.read_bytes()
    workload = json.loads(workload_bytes)
    if workload.get("phase") != "prefill" or sum(map(len, workload["groups"].values())) < 2:
        raise ValueError("expected grouped prefill prompts")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    meta = {
        "started_at": now(), "model": REPO, "revision": REV, "evidence": EVIDENCE,
        "phase": "prefill", "workload_id": workload["workload_id"],
        "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "provenance": workload["provenance"], "seed": workload["seed"],
        "prompt_records": [], "status": "FAIL",
    }
    try:
        import torch
        from vllm import LLM, SamplingParams, __version__
        if __version__ != "0.26.0":
            raise ValueError(f"instrumentation pinned to vLLM 0.26.0, got {__version__}")
        options = dict(dtype="half", max_model_len=256, max_num_batched_tokens=256,
                       max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False,
                       trust_remote_code=False, tensor_parallel_size=1,
                       kv_cache_memory_bytes=512 * 1024**2, seed=SEED)
        meta.update(vllm_version=__version__, torch_version=torch.__version__,
                    cuda=torch.version.cuda, engine_options=options)
        model_config = json.loads((args.snapshot / "config.json").read_bytes())
        meta["shared_expert_count_config"] = model_config.get("n_shared_experts")
        if meta["shared_expert_count_config"] != 2:
            raise ValueError("checkpoint config does not declare two shared experts")
        started = perf_counter()
        llm = LLM(model=str(args.snapshot.resolve(strict=True)), **options)
        meta.update(init_seconds=perf_counter() - started,
                    gpu=torch.cuda.get_device_name(), runtime_model_class=llm.apply_model(install_head_probe))
        tokenizer = llm.get_tokenizer()
        special = set(tokenizer.all_special_ids)
        records, all_rows = [], []
        sampling = SamplingParams(temperature=0.0, max_tokens=1, seed=SEED)
        layer_info = llm.apply_model(inventory)
        llm.apply_model(install_router_probe)
        shared_modules = llm.apply_model(install_shared_probe)
        for group, prompts in workload["groups"].items():
            for index, text in enumerate(prompts):
                sample_id = f"{group}-{index + 1:02d}"
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=True,
                    add_generation_prompt=True, return_dict=False)
                if not isinstance(ids, list) or not 1 <= len(ids) <= 192:
                    raise ValueError(f"prompt length outside 1..192: {sample_id}")
                request = {"prompt_token_ids": ids}
                # The baseline runs with the probe installed but its capture is discarded.
                llm.apply_model(reset_routes)
                baseline, = llm.generate([request], sampling, use_tqdm=False)
                baseline_output, = baseline.outputs
                before, = llm.apply_model(pop_head)
                # Discard baseline routes and shared calls before the traced request.
                llm.apply_model(reset_routes)
                traced, = llm.generate([request], sampling, use_tqdm=False)
                traced_output, = traced.outputs
                after, = llm.apply_model(pop_head)
                routes, = llm.apply_model(pop_routes)
                shared_routes, = llm.apply_model(pop_shared)
                if set(shared_routes) != set(range(1, 27)):
                    raise ValueError(f"shared expert calls missing for {sample_id}")
                if any(value["model_input_tokens"] != len(ids) for value in shared_routes.values()):
                    raise ValueError(f"shared expert token count differs from prompt for {sample_id}")
                ref, got = torch.tensor(before, dtype=torch.float32), torch.tensor(after, dtype=torch.float32)
                if (ref.shape != (1, 102400) or got.shape != ref.shape or not torch.equal(ref, got)
                        or list(baseline_output.token_ids) != list(traced_output.token_ids)):
                    raise ValueError(f"instrumented logits/output mismatch for {sample_id}")
                sample_dir = out / sample_id
                sample_dir.mkdir()
                write_json(sample_dir / "raw_route_capture.json", {"evidence": EVIDENCE, "layers": routes})
                from trace_awq_3090 import validate_and_export
                row_count = validate_and_export(routes, ids, special, sample_dir / "router_trace.jsonl")
                rows = [json.loads(line) for line in (sample_dir / "router_trace.jsonl").read_text(encoding="utf-8").splitlines()]
                for row in rows:
                    row.update(sample_id=sample_id, workload_group=group)
                all_rows.extend(rows)
                records.append({
                    "sample_id": sample_id, "workload_group": group, "prompt": text,
                    "prompt_ids": ids, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "prompt_tokens": len(ids), "valid_tokens": sum(t not in special for t in ids),
                    "generated_token_ids": list(traced_output.token_ids),
                    "logits_sha256": hashlib.sha256(ref.numpy().tobytes()).hexdigest(),
                    "logits_max_abs_difference": (ref-got).abs().max().item(),
                    "trace_records": row_count,
                    "shared_expert_modules": shared_modules,
                    "shared_expert_calls": shared_routes,
                })
        from p2.trace.router import write_trace
        write_trace(out / "router_trace.jsonl", all_rows, evidence=EVIDENCE)
        llm.apply_model(remove_shared_probe)
        meta.update(status="PASS", layers=layer_info, prompt_records=records,
                    record_count=len(all_rows), layer_ids=sorted({r["layer_id"] for r in all_rows}),
                    special_token_ids=sorted(special), trace="router_trace.jsonl")
        status = 0
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta.update(finished_at=now(), exit_code=locals().get("status", 1))
        write_json(out / "manifest.json", meta)


if __name__ == "__main__":
    main()
