# P2-O2 BF16 V1/UVA20 Continuous Decode — Self Check

Run ID: `OffloadGate-BF16-decode-20261120T172330`  
Date: 2026-11-20 (Asia/Shanghai)  
Status: **Task B PASS; Task C CPU simulation and independent audit PASS**

## Task B: real BF16 decode trace

The run used the official `deepseek-ai/DeepSeek-V2-Lite-Chat` revision `85864749cd611b4353ce1decdb286193298f64c7`, cached locally on the instance, vLLM 0.26.0, V1 `GPUModelRunner`, TP=1, physical GPU 0, BF16, UVA 20 GiB, 512 MiB KV cache, 256-token context, and the frozen `m0-4-curated-seed-v1` workload. No model download or conversion occurred. Runtime metadata reports `UVAOffloader` with 22,017,126,400 bytes offloaded and the Triton unquantized MoE path.

The 4-token smoke passed with identical baseline/instrumented token IDs, text, and finish reason. It was followed by six formal prompts, each producing 32 tokens and 31 continuous decode steps. Baseline/instrumented token IDs, text hashes, and finish reasons matched for all six. The manifest reports 4,550 prefill route rows, 4,836 decode rows, 26 layers, 64 experts, and native Top-6. The independently computed decode input SHA-256 is `dbe9920be8ecbfa8b6828e12d8c8921a53933c001b5ae02b001693b6f7a66993`.

Synchronized wall-clock means were 5.1435 s baseline and 6.1801 s instrumented for 32 output tokens, or 20.18% aggregate instrumentation overhead. This includes CPU copies, captured logits, and route validation. It is a six-request measurement and not an uninstrumented performance baseline. A 250 ms sampler recorded GPU0 peak 10,399 MiB and peak utilization 100% (527 samples); GPU1 was not used. After exit, both GPUs reported 0 MiB, both project locks were free, and the 2 GiB storage guard passed. Exit code was 0.

## Independent trace audit and precision boundary

The runner's initial layer inventory contained per-layer `quant_method`, router class, and monolithic-path status. The successful manifest later overwrote that object list with the integer layer IDs `[1..26]`; the per-layer `quant_method` inventory is therefore **not persisted in the final manifest**. The runner checked each layer was non-monolithic `UnquantizedFusedMoEMethod` before smoke/formal generation. Final evidence independently verifies the 26 layer IDs, full route-row coverage, V1 runner module, active UVA offloader, manifest invariants, Top-6 IDs, selected weights/probabilities, exact token alignment, row counts, and all output-file hashes. The manifest overwrite was retained and documented rather than filling in missing fields.

The independent auditor initially assumed `router_layers` held objects and stopped on the actual integer list. That schema mismatch was fixed in the isolated auditor; the full audit then passed. A separate CPU recheck found one BF16 softmax value differs across PyTorch builds at a rounding midpoint:

- Trace value: `general-01`, prefill, layer 25, sequence index 10, expert 21. Captured native weight and selected probability are both `0.029052734375`.
- Same 27×64 prefill batch under the instance's PyTorch 2.11.0+cu130 and the remote `checked_routes` implementation: BF16 candidate passes and returns `0.029052734375` exactly; the FP32 candidate rejects the BF16 native weight. CUDA remained uninitialized in this CPU-only reproduction.
- Same batch under local PyTorch 2.12.0+CPU: BF16 softmax returns `0.0289306640625`, one adjacent BF16 value lower. Maximum absolute cross-build difference is `0.0001220703125` (one BF16 ULP), in exactly one of 9,386 rows. The selected native weight equals the selected probability captured in the trace in every row. All native Top-6 sets meet the recomputed BF16 score cutoff; ties are accepted.
- Float64 softmax reference for that expert is `0.02899169904408871`; the midpoint between the two adjacent BF16 values is `0.02899169921875`, only `1.7466e-10` away. This is a numerical boundary case across CPU softmax implementations. The audit reports the one-ULP difference explicitly and does not claim cross-build exact softmax parity.

The code hash of remote and local `src/p2/trace/router.py` is `574385bbd00320b1b9a9d6eab175116df2997784d8d885aba0fd80ededfbe529`. The remote batch reproduction and local batch comparison are preserved in `remote-rounding-recheck.txt` and `local-rounding-recheck.txt`.

The original independent auditor also expected object-shaped layer inventory; it was updated only in this isolated run directory. The exact auditor uploaded before the run is preserved under `remote-inputs/` and matches the recorded remote input hash; the updated local audit source is `audit_bf16_decode_trace.py`. See these alongside `independent_audit.py` and `independent_audit.json`. No captured data was edited.

## Task C: independent cache simulation audit

The existing simulator used `decode_trace.jsonl` directly with capacities 6, 8, 12, 16, 32, and 64 experts per layer. Each expert was assigned 17,301,504 BF16 bytes (three 2048×1408 matrices, 2 bytes per value). Caches are per-layer and cold per request; each Top-6 set is simultaneous. The static-hot policy is leave-one-request-out and includes initial preload in its byte count.

The independent CPU implementation in `independent_audit.py` rehashed and reread the trace, then independently recomputed LRU, LFU, and static-hot hits/misses, every per-layer count, preload experts, and theoretical bytes for all 18 policy/capacity rows. All results matched `cache_simulation.json`; the access-frequency distribution also matched per layer. Aggregate results:

| Capacity | LRU hits / misses | LFU hits / misses | Static-hot hits / misses (preload experts) |
|---:|---:|---:|---:|
| 6 | 7,405 / 21,611 | 7,405 / 21,611 | 3,530 / 25,486 (936) |
| 8 | 8,685 / 20,331 | 9,121 / 19,895 | 4,645 / 24,371 (1,248) |
| 12 | 10,952 / 18,064 | 11,441 / 17,575 | 6,977 / 22,039 (1,872) |
| 16 | 12,974 / 16,042 | 13,329 / 15,687 | 9,167 / 19,849 (2,496) |
| 32 | 18,737 / 10,279 | 18,772 / 10,244 | 17,276 / 11,740 (4,992) |
| 64 | 21,136 / 7,880 | 21,136 / 7,880 | 28,963 / 53 (9,920) |

All byte totals are theoretical transfer counts based on the stated equal-sized-expert assumption; they are not PCIe measurements or latency predictions. Per-layer access histograms, hit/miss rows, and byte totals are in `cache_simulation.json` and `independent_audit.json`.

## File integrity and prior attempts

The remote SHA file covered 31 run files. Thirty matched directly. `bootstrap.log` continued receiving tee output after the runner generated `sha256.txt`; an after-exit remote rehash of the final log also matched its local copy. The workload SHA-256 is `c934f433108052632a90b313e1e12a6f04828c67f7c06f67e0a00a28d37319f0`. Raw remote output, logs, environment snapshots, sampler, manifest, and hashes are under `remote-evidence/`.

Earlier failed attempts remain in their own directories and archives, including the external TP=2 overlap OOM and the first smoke hook `NameError`. The hook correction for this successful run was only a module-level `import torch`; it passed AST and CPU import/closure checks without initializing CUDA. Task A's prior TP=1/UVA20 9/9 baseline was not repeated. No shared PLAN/DECISIONS/LEDGER file was changed and no Git commit was made.

