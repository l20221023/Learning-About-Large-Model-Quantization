#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/autodl_storage_env.sh"
WORK="$P2_WORK_ROOT"
SNAPSHOT="$P2_MODELS_DIR/TechxGenus-DeepSeek-V2-Lite-Chat-AWQ-e9fdf80"
WORKLOAD="$WORK/workloads/m0_5_seed_prompts.json"
PYTHON="$WORK/venv-vllm-0.26.0/bin/python"
RUN_ID="M0-5-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-3090"
OUT="$P2_EXPERIMENTS_DIR/$RUN_ID"
LOG="$P2_LOGS_DIR/$RUN_ID"
mkdir -p "$OUT" "$LOG"
if [[ ! -x "$PYTHON" || ! -d "$SNAPSHOT" || ! -f "$WORKLOAD" ]]; then
    echo 'missing existing model, workload, or isolated runtime' >&2
    exit 66
fi
"$SCRIPT_DIR/autodl_storage_guard.sh" check 1073741824
gpu_monitor=''
cleanup() { [[ -n $gpu_monitor ]] && kill "$gpu_monitor" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
set +e
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 250 > "$LOG/gpu.csv" 2>&1 &
gpu_monitor=$!
"$SCRIPT_DIR/autodl_storage_guard.sh" run 1073741824 -- env \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    PATH="$WORK/venv-vllm-0.26.0/bin:$PATH" \
    "$PYTHON" "$WORK/profile_awq_m0_5.py" --snapshot "$SNAPSHOT" \
    --workload "$WORKLOAD" --output-dir "$OUT/raw" > "$LOG/run.log" 2>&1
status=$?
cleanup
wait "$gpu_monitor" 2>/dev/null
set -e
printf '%s\n' "$status" > "$LOG/exit"
printf 'run_id=%s\noutput=%s\nlogs=%s\nexit=%s\n' "$RUN_ID" "$OUT" "$LOG" "$status"
exit "$status"
