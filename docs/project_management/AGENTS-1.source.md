# Codex Repository Instructions — LLM Quantization & MoE Optimization

## Persistent context
- Read `project_status.md`, the applicable task plan, and existing evidence before changing code or running experiments.
- Main model currently: `deepseek-ai/DeepSeek-V2-Lite-Chat`. Historical conversation title “32B” does not change the current experiment subject.
- Keep B0_TP2 frozen: 2x RTX 3090, BF16, TP=2, vLLM 0.26.0, max_model_len=2304, max_num_seqs=8. Its 2400 formal requests are independently audited. Do not overwrite baseline datasets, results or audit files.
- Workstreams: O1 single-GPU BF16 V1+UVA Offloading; Q1 formal BF16 quality reference; O2 Router Trace and cache simulation; U1 UE replacement evaluation; J1 combined optimization.

## Operating rules
1. Only do the task explicitly assigned to this Codex thread. Do not start INT4, UE training, Offloading changes or Router instrumentation while assigned only Q1.
2. Before tests, check GPU availability, output disk free space, existing running jobs and service health. Never terminate another workstream's service, change model weights, or launch a second memory-heavy model without approval.
3. Fix runnable problems when safe; do not end after a smoke test if the full assigned task remains feasible. Mark BLOCKED only with command, exit status, logs and a concrete dependency.
4. For every experimental claim preserve: exact model/source revision, environment, inputs, seeds, command/script, sample-level records, summary methodology and verification results.
5. Treat HTTP text chunks as transport chunks, not one-token ITL. Distinguish observed GPU memory samples from actual instantaneous peaks. Disclose prefix cache and hardware/TP differences.
6. Do not call 24-question heuristic output checks a formal capability benchmark. Keep evaluation subsets and scoring protocols fixed.
7. Do not edit `project_status.md` unless designated the status integrator. All other threads read it and write their task-specific status under `task_status/<task_id>.md`, with evidence paths and commit hash. The integrator validates and merges updates into `project_status.md`.
8. Do not modify unrelated worktrees, discard others' changes, overwrite frozen data, or commit model weights, credentials, caches or very large raw outputs.

## Completion rules
- `COMPLETED` requires actual deliverables and successful independent checks.
- `PARTIAL` means useful validated work exists but acceptance criteria remain unmet.
- `WAITING` means not yet started or intentionally deferred.
- `BLOCKED` requires a demonstrated hard blocker and sufficient diagnostics.
- Report exact artifact paths, checks, unresolved limits and next gated step.
