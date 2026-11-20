# LLM Quantization & MoE Optimization — Shared Project Status

> Status authority: this file is the reviewed project summary, **not a substitute for raw evidence**.
> Status integrator: **designate one Codex task / maintainer before writing**.
> Last verified update: 2026-11-20 (initial template; verify against repository before accepting).

## Objective
Study compression, BF16 expert offloading, MoE routing and United Expert substitution on DeepSeek-V2-Lite-Chat. Preserve causal comparability and distinguish engineering utility from paper novelty.

## Workstreams

| ID | Owner | Status | Evidence | Next acceptance gate |
| --- | --- | --- | --- | --- |
| B0_TP2 | Baseline / reviewed | COMPLETED | `benchmarks/deepseek_v2_lite/reports/baseline_report.md`; `benchmarks/deepseek_v2_lite/results/B0_TP2/audit.json` (confirm actual repo paths) | Archive; do not rerun by default |
| O1 | Assign Offloading Codex | PARTIAL | V1+UVA single-3090 short-window forward / causal validation report; locate in repository | Continuous Decode, explicit numerical tolerances, performance/stability |
| Q1 | Assign Quality Codex | PARTIAL | Dataset mirror viability established; formal frozen 256-sample runs not yet verified | Freeze GSM8K/C-Eval/MMLU samples and scoring; run + audit |
| O2 | Not yet assigned | WAITING | MoE execution-path notes if available | Real Router Trace and offline cache simulation |
| U1 | Not yet assigned | NEEDS_REVIEW | Existing UE code/experiments may exist; inspect before assigning status | Audit real implementation, replacement cost and quality loss |
| J1 | Not yet assigned | WAITING | — | Comparable combined-policy tests |

## Immutable Baseline Contract
- Original reference: BF16, TP=2, 2×RTX 3090, vLLM 0.26.0, max_model_len=2304, max_num_seqs=8.
- P01–P08, 3 repeats × 100 formal requests = 2400; frozen 300 prompt samples with exact post-template input lengths 128/512/2048.
- ITL = N/A; stream chunks are not token timestamps. Prefix caching and resource sample limitations disclosed in report.
- The 24-question heuristic quality snapshot (18/24 simple checks) is not an official model capability score.

## Coordination and update protocol
1. Every agent reads this file before starting and syncs its repository checkout/branch when applicable.
2. **Only the designated integrator edits this summary**. Other agents write `task_status/Q1.md`, `task_status/O1.md`, etc. and commit/push those notes when using separate clones/worktrees.
3. The integrator checks evidence, commit hash and acceptance criteria before changing statuses. Never use `COMPLETED` from a self-report alone.
4. If branches diverge, resolve through Git review or a pull request; do not force push or overwrite unrelated changes.
5. Raw weights, caches and large benchmark logs stay on designated disks; store lightweight manifest, hashes, summaries and paths in Git.

## Open decisions
- Identify the status integrator and exact repository remote/branch.
- Confirm both Codex threads' working directories and Git worktrees; do not assume identical paths imply shared files.
- Confirm whether O1 and Q1 share the same physical GPUs and schedule tests to avoid interference.
