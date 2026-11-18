#!/usr/bin/env python3
"""Compute raw-text Wikitext-2 test PPL with vLLM prompt_logprobs."""
import argparse, hashlib, io, json, math, time, traceback, zipfile
from pathlib import Path

DATA_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
MODEL_REVISION = "e9fdf80ec6f1d200a029c41004bee40993763b43"
WINDOW, STRIDE = 256, 128


def windows(token_ids, window=WINDOW, stride=STRIDE):
    """Return (context IDs, local target positions); score each causal target once."""
    if window < 2 or not 0 < stride < window:
        raise ValueError("require window >= 2 and 0 < stride < window")
    result, next_target = [], 1
    for start in range(0, len(token_ids), stride):
        end = min(start + window, len(token_ids))
        if end - start < 2:
            break
        first_global = max(start + 1, next_target)
        if first_global < end:
            result.append((start, token_ids[start:end], list(range(first_global-start, end-start))))
            next_target = end
        if end == len(token_ids):
            break
    return result


def extract_target_logprobs(prompt_logprobs, target_positions, token_ids):
    scores=[]
    for pos in target_positions:
        token=token_ids[pos]
        if prompt_logprobs is None or pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            raise ValueError(f"missing prompt logprob at token position {pos}")
        row=prompt_logprobs[pos]
        entry=row.get(token) if hasattr(row,"get") else None
        if entry is None:
            raise ValueError(f"target token {token} absent from returned logprobs at position {pos}")
        value=getattr(entry,"logprob",None)
        if value is None and isinstance(entry,dict): value=entry.get("logprob")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"invalid target logprob at position {pos}: {value}")
        scores.append(float(value))
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True); p.add_argument("--data-zip")
    p.add_argument("--data-parquet")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--max-windows",type=int,default=0); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False)
    m={"status":"FAIL","model_revision":MODEL_REVISION,"data_sha256_expected":DATA_SHA256,
       "window":WINDOW,"stride":STRIDE,"started_unix":time.time()}
    try:
        if bool(a.data_zip) == bool(a.data_parquet):
            raise ValueError("provide exactly one of --data-zip and --data-parquet")
        if a.data_zip:
            zip_bytes=Path(a.data_zip).read_bytes(); m["zip_sha256_actual"]=hashlib.sha256(zip_bytes).hexdigest()
            if m["zip_sha256_actual"] != DATA_SHA256: raise ValueError("dataset ZIP SHA-256 mismatch")
            member="wikitext-2-raw/wiki.test.raw"
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names=[x for x in zf.namelist() if x==member]
                if len(names)!=1: raise ValueError(f"expected exactly one ZIP member {member!r}")
                raw=zf.read(member)
        else:
            import pyarrow.parquet as pq
            parquet_bytes=Path(a.data_parquet).read_bytes()
            expected="5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
            m["parquet_sha256_actual"]=hashlib.sha256(parquet_bytes).hexdigest()
            if m["parquet_sha256_actual"] != expected: raise ValueError("Parquet SHA-256 mismatch")
            table=pq.read_table(a.data_parquet)
            if table.column_names != ["text"] or table.num_rows != 4358: raise ValueError("unexpected Parquet schema or rows")
            rows=table.column("text").to_pylist()
            if any(not isinstance(row,str) for row in rows): raise ValueError("non-string Parquet row")
            raw="".join(row if row.endswith("\n") else row+"\n" for row in rows).encode("utf-8")
            member="Salesforce/wikitext@b08601e/wikitext-2-raw-v1/test.parquet; newline-normalized stream"
        text=raw.decode("utf-8",errors="strict")
        m.update(test_member=member,test_bytes=len(raw),test_sha256=hashlib.sha256(raw).hexdigest(),
                 test_rows=len(text.splitlines()))
        import torch, vllm
        from vllm import LLM, SamplingParams
        if vllm.__version__!="0.26.0": raise ValueError(f"expected vLLM 0.26.0, got {vllm.__version__}")
        opts=dict(dtype="half",max_model_len=257,max_num_batched_tokens=257,max_num_seqs=1,
                  enforce_eager=True,enable_prefix_caching=False,trust_remote_code=False,
                  tensor_parallel_size=1,kv_cache_memory_bytes=512*1024**2,seed=20261117)
        m.update(vllm=vllm.__version__,torch=torch.__version__,cuda=torch.version.cuda,
                 engine_options=opts,snapshot=str(Path(a.snapshot).resolve()))
        total_started=time.perf_counter(); llm=LLM(model=a.snapshot,**opts); tokenizer=llm.get_tokenizer()
        params=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1)
        ids=tokenizer.encode(text,add_special_tokens=False)
        m["tokenized_ids_sha256"]=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
        raw_lines=[]; total_nll=0.; count=0; ppl_started=time.perf_counter()
        for win_i,(start,ctx,target_pos) in enumerate(windows(ids)):
            if a.max_windows and win_i >= a.max_windows: break
            result,=llm.generate([{"prompt_token_ids":ctx}],params,use_tqdm=False)
            scores=extract_target_logprobs(result.prompt_logprobs,target_pos,ctx)
            total_nll-=sum(scores); count+=len(scores)
            raw_lines.append(json.dumps({"window":win_i,"input_tokens":len(ctx),
                "global_start":start,"target_positions":target_pos,
                "target_token_ids":[ctx[i] for i in target_pos],"target_logprobs":scores},separators=(",",":")))
        expected_count = min(max(0,len(ids)-1), WINDOW + max(0,a.max_windows-1)*STRIDE-1) if a.max_windows else max(0,len(ids)-1)
        if count != expected_count: raise ValueError(f"target coverage {count} != expected ({expected_count})")
        (out/"raw_scores.jsonl").write_text("\n".join(raw_lines)+"\n")
        ppl_elapsed=time.perf_counter()-ppl_started
        m.update(status="SMOKE_PASS" if a.max_windows else "PASS",token_count=len(ids),scored_tokens=count,nll_sum=total_nll,mean_nll=total_nll/count,
                 perplexity=math.exp(total_nll/count),ppl_elapsed_s=ppl_elapsed,
                 elapsed_s=time.perf_counter()-total_started,
                 raw_sha256=hashlib.sha256((out/"raw_scores.jsonl").read_bytes()).hexdigest())
    except Exception as e: m.update(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
    finally:
        m["finished_unix"]=time.time(); (out/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    if m["status"] not in ("PASS","SMOKE_PASS"): raise SystemExit(1)
    print(json.dumps(m,indent=2))

if __name__=="__main__": main()
