#!/usr/bin/env python3
"""Minimal vLLM 0.26.0 in-process AWQ decode probe; no benchmark claims."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

REPO = "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ"
REV = "e9fdf80ec6f1d200a029c41004bee40993763b43"
PROMPT = "Explain what an expert router does in one sentence."
SEED = 20261117


def atomic_json(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extend-64", action="store_true", help="run only after max_tokens=2 passes")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    options = dict(dtype="half", max_model_len=256, max_num_batched_tokens=256,
                   max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False,
                   trust_remote_code=False, tensor_parallel_size=1,
                   kv_cache_memory_bytes=512 * 1024**2, seed=SEED)
    result = {"model": REPO, "revision": REV, "prompt": PROMPT,
              "options": options, "status": "FAIL", "steps": [],
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        from vllm import LLM, SamplingParams, __version__
        result.update(vllm_version=__version__, torch_version=torch.__version__,
                      cuda_version=torch.version.cuda)
        if __version__ != "0.26.0":
            raise ValueError("probe pinned to vLLM 0.26.0")
        snapshot = args.snapshot.resolve(strict=True)
        result["snapshot"] = str(snapshot)
        result["quantization_config"] = json.loads((snapshot / "config.json").read_text())["quantization_config"]
        started = time.perf_counter()
        llm = LLM(model=str(snapshot), **options)
        result["init_s"] = time.perf_counter() - started
        result["gpu"] = torch.cuda.get_device_name()
        tokenizer = llm.get_tokenizer()
        ids = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                            tokenize=True, add_generation_prompt=True,
                                            return_dict=False)
        if not isinstance(ids, list) or not 1 <= len(ids) <= 64:
            raise ValueError("unexpected prompt token count")
        result["prompt_token_ids"] = ids
        for count in ([2, 64] if args.extend_64 else [2]):
            started = time.perf_counter()
            response, = llm.generate([{"prompt_token_ids": ids}],
                                     SamplingParams(temperature=0.0, max_tokens=count,
                                                    min_tokens=count, seed=SEED), use_tqdm=False)
            item, = response.outputs
            token_ids = list(item.token_ids)
            step = {"max_tokens": count, "generated_token_ids": token_ids,
                    "generated_text": item.text, "finish_reason": item.finish_reason,
                    "elapsed_s": time.perf_counter() - started}
            result["steps"].append(step)
            atomic_json(out / f"step-{count}.json", step)
            if len(token_ids) != count:
                raise ValueError(f"max_tokens={count}: generated {len(token_ids)} tokens")
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(out / "manifest.json", result)


if __name__ == "__main__":
    main()
