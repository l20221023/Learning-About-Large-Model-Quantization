# M0-4 Routing, Load and Coactivation Self-Check

**Status: PASS for a small exploratory community-AWQ prefill workload.** This does not establish production traffic distributions, BF16 routing, expert pruning safety, or general task behavior.

## 1. Scope and source

The experiment used the already downloaded `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ` checkpoint at revision `e9fdf80ec6f1d200a029c41004bee40993763b43`, vLLM 0.26.0, FP16 router softmax and the existing single RTX 3090. No weights were downloaded or changed during M0-4.

The workload in [m0_4_seed_prompts.json](../workloads/m0_4_seed_prompts.json) has two operator-authored prompts in each of general English, Chinese and code. It contains no external corpus or user data. Prompts are tokenized using the checkpoint's Chat Template. This is a small curated seed set, not a representative service distribution. We captured prefill routing. A one-token greedy output was requested only to preserve the existing full-logits instrumentation check; decode routing is not part of these counts.

## 2. Captured data

The six prompts contained 169 non-special content tokens: 47 general, 50 Chinese and 72 code. Their full templated lengths, including special tokens, were respectively 27 and 22, 24 and 28, and 34 and 40. The run captured every valid token at all 26 routed MoE layers, for 4,394 rows. Each row keeps all 64 router logits and probabilities, the native six expert IDs and selected weights, plus sample and workload labels.

The model config declares two shared experts. A read-only forward hook found the `DeepseekV2MLP` shared module called once per prompt at each routed layer. Across the 156 sample-layer pairs, the hook input covered each prompt's full template token length. The shared activity CSV distinguishes valid content tokens from special template tokens.

## 3. Checks and results

- All 26 MoE layers and all six prompts were present; the dense layer 0 was excluded.
- For each group and layer, selected expert counts summed to valid content tokens × 6. This was 282 general, 300 Chinese and 432 code selections per layer.
- For each group and layer, the upper triangle of the 64×64 coactivation matrix summed to valid content tokens × 15. The matrix is symmetric; its diagonal equals per-expert selection counts.
- Every selected ID is unique and in 0–63. Native weights match the high-precision softmax of captured FP16 logits within the corresponding FP16 rounding interval. This handles CUDA/CPU rounding near half-precision midpoints without widening to arbitrary numeric tolerance.
- Six baseline-versus-instrumented comparisons had exact equality for the complete 102,400-value output logits and generated token IDs.
- Shared module coverage was 26/26 for all six prompts; every module call count was 1 and its input-token count matched the full templated prompt length.
- Independent trace verification passed. 27 local M0-2/M0-3/M0-4 unit tests passed.

Across the 26 layer summaries, mean routed-expert load CV was 0.875 for general, 0.976 for Chinese and 0.736 for code. These values describe only this tiny curated sample; they are not estimates of live traffic. The range across layers is retained in `load_summary.csv`, and per-prompt distributions are retained in `sample_load_summary.csv` and `sample_routing_counts.csv`.

The run lasted 25.8 seconds from manifest start to finish. The 250 ms GPU sampler recorded 104 samples, a peak of 10,595 MiB and maximum sampled utilization of 13%. This is the peak for this run and sampling interval, not a hardware-wide maximum.

## 4. Artifacts

The self-contained result directory is `experiments/results/M0-4-20261117T235424-3090/`.

- `payload/M0-4-20261117T235424-3090/trace/`: consolidated `router_trace.jsonl`, manifest, and each prompt's raw route capture and individual trace.
- `analysis/routing_counts.csv`: group/layer/expert selection counts and mean probabilities/selected weights.
- `analysis/coactivation.csv`: 64×64 matrices by workload group and layer.
- `analysis/load_summary.csv` and `sample_load_summary.csv`: layer-level and prompt-level load skew.
- `analysis/sample_routing_counts.csv`: per-prompt expert counts and mean probabilities.
- `analysis/shared_expert_activity.csv`: per-prompt/layer Shared module calls and input tokens.
- `analysis/layer_charts/`: one SVG for each of the 26 routed layers.
- `payload/.../run.log`, `gpu.csv`, `exit`: runtime log, GPU sampling and exit status.
- `M0-4-20261117T235424-3090.tar.gz`: original cloud archive, SHA-256 `d6a604207117c34390fcc00b70c834ad49f748877a20e9a2831d0a948ef55d20`. It also preserves earlier failed attempts and storage-audit evidence.

## 5. Issues and repairs

The first attempts exposed deployment-path errors (`src/p2` and workload path), then a CPU FP16 softmax midpoint mismatch and baseline requests being included in the Router hook. Each failure log and available raw capture is preserved in the evidence archive. The export now resolves the deployed source tree, discards baseline hook state, and checks FP16 values against the exact softmax rounding interval. The same interval verifier passed all 52 layer/prompt raw captures collected during diagnosis before the final full run.

## 6. Reproduction

On the same cloud instance, with the existing model and vLLM environment:

```bash
bash /root/autodl-tmp/p2-deepseek-moe/run_m0_4_autodl.sh
```

Locally, verify and aggregate the archived trace:

```powershell
$py = 'C:\Users\li\.conda\envs\pytorch_clean\python.exe'
$trace = 'experiments/results/M0-4-20261117T235424-3090/payload/M0-4-20261117T235424-3090/trace'
& $py -B scripts/verify_m0_4.py $trace --workload workloads/m0_4_seed_prompts.json
& $py -B scripts/aggregate_m0_4.py "$trace/router_trace.jsonl" --out experiments/results/M0-4-20261117T235424-3090/analysis --manifest "$trace/manifest.json"
& $py -B -m unittest discover -s tests -p 'test_m0_*.py'
```

## 7. Limits and next step

The official BF16 checkpoint was not downloaded or compared. AWQ routing can differ from BF16. Only two prompts per category and prefill positions were sampled; confidence intervals and live-traffic generalization are unavailable. No conclusion about expert deletion or merging follows from these counts.

Per the user's instruction, M0-3 and M0-4 self-check reports are being accumulated for unified acceptance after subsequent experiments. Continue to M0-5 using the existing AWQ checkpoint and the data-volume storage guard; do not infer BF16 equivalence.
