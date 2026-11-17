"""Single-request, offline AWQ smoke and router probe for vLLM 0.26.0."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2.weights.audit import write_json
from p2.weights.safetensors_io import now

REPO = "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ"
REV = "e9fdf80ec6f1d200a029c41004bee40993763b43"
PROMPT = "Explain what an expert router does in a mixture-of-experts model in one sentence."
EVIDENCE = "MEASURED_COMMUNITY_AWQ_INT4"
SEED = 20261117


def install_head_probe(model):
    original = model.compute_logits
    model._p2_head = []

    def compute_logits(*args, **kwargs):
        value = original(*args, **kwargs)
        if value is not None:
            model._p2_head.append(value.detach().float().cpu())
        return value

    model.compute_logits = compute_logits
    return type(model).__name__


def inventory(model):
    layers = model.model.layers
    if len(layers) != 27 or hasattr(layers[0].mlp, "experts"):
        raise ValueError("unexpected decoder topology")
    details = []
    for layer, block in enumerate(layers[1:], start=1):
        runner = block.mlp.experts
        quant = runner.routed_experts.quant_method
        kernel = getattr(quant, "moe_kernel", None)
        experts = getattr(getattr(kernel, "impl", None), "fused_experts", None)
        details.append({
            "layer_id": layer, "attention": type(block.self_attn).__name__,
            "router": type(runner.router).__name__, "moe_quant_method": type(quant).__name__,
            "monolithic": quant.is_monolithic,
            "moe_kernel": type(kernel).__name__ if kernel is not None else None,
            "moe_kernel_experts": type(experts).__name__ if experts is not None else None,
        })
    return details


def install_router_probe(model):
    details = inventory(model)
    if any(row["monolithic"] for row in details):
        raise ValueError("selected MoE kernel routes inside a monolithic operation")
    model._p2_routes = {}

    def wrap(layer, original):
        def select(*args, **kwargs):
            weights, ids = original(*args, **kwargs)
            logits = kwargs["router_logits"]
            if layer in model._p2_routes:
                raise ValueError(f"unexpected repeated router call for layer {layer}")
            tensors = tuple(tensor.detach().cpu().tolist() for tensor in (logits, ids, weights))
            model._p2_routes[layer] = (str(logits.dtype), *tensors)
            return weights, ids
        return select

    for layer, block in enumerate(model.model.layers[1:], start=1):
        router = block.mlp.experts.router
        router.select_experts = wrap(layer, router.select_experts)
    return len(details)


def pop_head(model):
    if len(model._p2_head) != 1:
        raise ValueError(f"expected one logits call, observed {len(model._p2_head)}")
    return model._p2_head.pop().tolist()


def pop_routes(model):
    return model._p2_routes


def validate_and_export(routes, prompt_ids, special_ids, output, *, evidence=EVIDENCE):
    import torch

    from p2.trace.router import checked_routes, write_trace

    expected = set(range(1, 27))
    if routes.keys() != expected:
        raise ValueError(f"routed layer set differs: {sorted(routes)}")
    rows = []
    for layer in sorted(routes):
        dtype, logits, ids, weights = routes[layer]
        if len(logits) != len(prompt_ids):
            raise ValueError(f"layer {layer} router token count differs from prompt")
        softmax_dtype = {"torch.float16": torch.float16, "torch.float32": torch.float32}[dtype]
        probs = checked_routes(torch.tensor(logits, dtype=softmax_dtype),
                               torch.tensor(ids, dtype=torch.long),
                               torch.tensor(weights, dtype=torch.float32), softmax_dtype=softmax_dtype).tolist()
        for position, token in enumerate(prompt_ids):
            if token in special_ids:
                continue
            rows.append({
                "layer_id": layer, "batch_id": 0, "sequence_index": position, "token_id": token,
                "router_logits": logits[position], "router_probabilities": probs[position],
                "topk_expert_ids": ids[position], "topk_weights": weights[position],
                "router_softmax_dtype": dtype,
            })
    write_trace(output, rows, evidence=evidence)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    options = dict(
        dtype="half", max_model_len=256, max_num_batched_tokens=256, max_num_seqs=1,
        enforce_eager=True, enable_prefix_caching=False, trust_remote_code=False,
        tensor_parallel_size=1, kv_cache_memory_bytes=512 * 1024**2, seed=SEED,
    )
    metadata = dict(
        started_at=now(), model=REPO, revision=REV, evidence=EVIDENCE,
        phase="prefill", engine_options=options, status="FAIL",
    )
    status = 1
    try:
        import torch
        from vllm import LLM, SamplingParams, __version__

        metadata.update(vllm_version=__version__, torch_version=torch.__version__, cuda=torch.version.cuda)
        if __version__ != "0.26.0":
            raise ValueError("this probe's native instrumentation is pinned to vLLM 0.26.0")
        path = args.snapshot.resolve(strict=True)
        config = json.loads((path / "config.json").read_bytes())
        metadata["quantization_config"] = config["quantization_config"]
        started = perf_counter()
        llm = LLM(model=str(path), **options)
        metadata.update(init_seconds=perf_counter() - started, gpu=torch.cuda.get_device_name())
        tokenizer = llm.get_tokenizer()
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=True, add_generation_prompt=True,
            return_dict=False,
        )
        if not isinstance(prompt_ids, list) or not 1 <= len(prompt_ids) <= 64:
            raise ValueError("expected at most 64 prompt token IDs")
        special = set(tokenizer.all_special_ids)
        prompt = {"prompt_token_ids": prompt_ids}
        sampling = SamplingParams(temperature=0.0, max_tokens=1, seed=SEED)
        started = perf_counter()
        if not args.smoke_only:
            metadata["runtime_model_class"], = llm.apply_model(install_head_probe)
        baseline, = llm.generate([prompt], sampling, use_tqdm=False)
        output, = baseline.outputs
        if len(output.token_ids) != 1:
            raise ValueError("short request produced no output token")
        metadata.update(
            prompt=PROMPT, prompt_token_ids=prompt_ids, special_token_ids=sorted(special),
            generated_token_ids=list(output.token_ids), generated_text=output.text,
            smoke_seconds=perf_counter() - started, status="SMOKE_PASS",
        )
        write_json(out / "smoke.json", metadata)
        metadata["layers"], = llm.apply_model(inventory)
        if not args.smoke_only:
            before, = llm.apply_model(pop_head)
            llm.apply_model(install_router_probe)
            traced, = llm.generate([prompt], sampling, use_tqdm=False)
            after, = llm.apply_model(pop_head)
            routes, = llm.apply_model(pop_routes)
            reference = torch.tensor(before, dtype=torch.float32)
            actual = torch.tensor(after, dtype=torch.float32)
            write_json(out / "raw_route_capture.json", {"evidence": EVIDENCE, "layers": routes})
            if (reference.shape != (1, 102400) or reference.shape != actual.shape
                    or not torch.isfinite(reference).all() or not torch.equal(reference, actual)
                    or list(traced.outputs[0].token_ids) != list(output.token_ids)):
                raise ValueError("instrumented request did not reproduce baseline output logits")
            count = validate_and_export(routes, prompt_ids, special, out / "router_trace.jsonl")
            metadata.update(
                status="PASS", trace="router_trace.jsonl",
                inputs={"input_ids": [prompt_ids], "attention_mask": [[1] * len(prompt_ids)]},
                excluded_token_ids=sorted(special),
                logits_sha256=hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
                router_softmax_dtypes=sorted({capture[0] for capture in routes.values()}),
                metrics={
                    "logits_exact_equal": True,
                    "logits_max_abs_difference": (reference - actual).abs().max().item(),
                    "layer_ids": sorted(routes),
                    "valid_tokens": sum(token not in special for token in prompt_ids),
                    "records": count,
                },
            )
        status = 0
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata.update(finished_at=now(), exit_code=status)
        write_json(out / "manifest.json", metadata)


if __name__ == "__main__":
    main()
