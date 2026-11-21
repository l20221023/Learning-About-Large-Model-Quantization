# P2-M1 Profiling risks and M3 plan (initial audit)

- **Service restart:** modifying installed vLLM Python/kernel code requires a service restart to load it. Do not restart during P2-M1. Use a copy-on-write code overlay or separate checkout for M3 and schedule the restart explicitly.
- **Second model:** avoid loading a second BF16 copy on the same two 24-GiB GPUs. Minimal router unit tests can use CPU tensors and the installed Python routing logic where supported; a production-path validation should reuse the single service in a planned maintenance window.
- **CUDA Graph / fused path:** the B0 and Q1 service logs say `enforce_eager=True` and `cudagraph_mode=NONE`, so graph replay interference was not present in these observed runs. `moe_backend='auto'` selected fused CUDA kernels; a Python hook around experts will not reveal per-expert outputs and an observer can force graph breaks if settings change later.
- **Synchronization:** copying GPU IDs/weights to CPU inside every layer/token step can synchronize the stream and perturb scheduling. Keep bounded device buffers, enqueue nonblocking copies on an owned stream with events, drain asynchronously, and measure dropped-buffer counters. Do not call `.item()`, `.cpu()` or `torch.cuda.synchronize()` on the hot path.
- **Padding/duplicates:** scheduler batches can include padding, speculative/invalid positions, or repeated execution due to retries. Carry explicit validity and batch/step IDs; deduplicate only by a defined `(trace_session, step_id, layer_id, token_row)` key. Do not deduplicate identical token text.
- **Trace size:** dense upper bound is 26 MoE layers × 6 IDs per generated/prefill token; use bounded sampling or a fixed token budget and record every drop. Keep raw traces separate from summaries.

## Minimum M3 execution sequence

1. Freeze current server/model/B0 hashes and confirm both GPUs and disk reserves; preserve current B0 files untouched.
2. In a separate vLLM 0.26.0 source overlay, add an opt-in router observer after runtime backend selection, plus validity/row-map propagation and bounded device buffers.
3. Run CPU unit tests for score/top-k/ID mapping and malformed/padded inputs. Run `python -m compileall` and a clean source diff review.
4. In a maintenance window, restart the existing TP=2 service from the overlay (no second model); use a tiny fixed prompt and one request first.
5. Verify IDs/weights at router boundary against dispatch counts and ensure exactly six valid selections per valid token per MoE layer. Compare pre/post logits on the same fixed inputs, report dropped rows and latency overhead, and preserve raw trace/hash.
6. Only then run the planned trace workload. User request forbids service restart in P2-M1, so steps 4–6 were not executed.

## Current observation

The historical B0 server process shut down at 2026-11-20 16:12:13 host log time; port 8000 now refuses connections, there are no vLLM processes or listeners, and both GPUs report 0 MiB. Do not treat the old startup log as live service state. This task does not authorize restarting the service; M3 must first be explicitly scheduled with an approved startup window and unchanged model/config hashes.
