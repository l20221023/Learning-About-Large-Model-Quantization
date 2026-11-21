# Reproduction commands

Remote workspace: `/root/autodl-tmp/p2-m3-routertrace-20261120`; cached model: `/root/autodl-tmp/p2-deepseek-moe/models/DeepSeek-V2-Lite-Chat-BF16-85864749`.

Input preparation (CPU only):

```bash
source /root/autodl-tmp/p2-deepseek-moe/venv-vllm-0.26.0/bin/activate
python /root/autodl-tmp/p2-m3-routertrace-20261120/prepare_workload.py
```

Trace ON and OFF used the same storage guard and paired server arguments; only the capture flag differs:

```bash
WORK=/root/autodl-tmp/p2-deepseek-moe
"$WORK/autodl_storage_guard.sh" run 2147483648 -- bash /root/autodl-tmp/p2-m3-routertrace-20261120/run_mode_autodl.sh on 03
"$WORK/autodl_storage_guard.sh" run 2147483648 -- bash /root/autodl-tmp/p2-m3-routertrace-20261120/run_mode_autodl.sh off 01
```

The wrapper took both project GPU locks, checked both GPUs, storage reserve, and port 8000, then launched the existing BF16 snapshot with vLLM 0.26.0, TP=2, eager, 2304 context, eight max sequences, and TritonExperts. It set `VLLM_USE_V2_MODEL_RUNNER=0` to select V1 and added `--enable-return-routed-experts` only for ON. It ran one smoke request first; the formal six-prompt collection ran only after the smoke route shape/mapping/ID checks passed. Each mode included two repeated prompt cache probes after the formal set.

Independent audit:

```bash
python /root/autodl-tmp/p2-m3-routertrace-20261120/audit_m3.py \
  --evidence-root /root/autodl-tmp/experiments/P2-M3
```

The formal exit codes are in `remote-evidence/on-03/formal.exit` and `remote-evidence/off-01/formal.exit`; both are 0. Prior failed preparation, startup-gate, smoke-report, and audit attempts remain under `attempts.md`, `preparation_attempts.md`, `remote-evidence/`, and `remote-runs/`.
