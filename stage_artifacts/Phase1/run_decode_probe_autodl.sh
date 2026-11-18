#!/usr/bin/env bash
set -euo pipefail
WORK=/root/autodl-tmp/p2-deepseek-moe
source "$WORK/autodl_storage_env.sh"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY="$WORK/venv-vllm-0.26.0/bin/python"
SNAPSHOT="$P2_MODELS_DIR/TechxGenus-DeepSeek-V2-Lite-Chat-AWQ-e9fdf80"
RUN_ID="Phase1-decode-probe-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-3090"
OUT="$P2_EXPERIMENTS_DIR/$RUN_ID"
mkdir -p "$OUT"
[[ -x $PY && -d $SNAPSHOT ]] || { echo 'missing existing Python or AWQ snapshot' >&2; exit 66; }
extra=()
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --extend-64 ) ]]; then
  echo 'usage: run_decode_probe_autodl.sh [--extend-64]' >&2
  exit 64
fi
[[ $# -eq 1 ]] && extra+=(--extend-64)
sha256sum "$HERE/probe_awq_inprocess_decode.py" > "$OUT/input-sha256.txt"
df -B1 / /root/autodl-tmp > "$OUT/df-before.txt"
nvidia-smi > "$OUT/gpu-before.txt"
"$WORK/autodl_storage_guard.sh" check 268435456 > "$OUT/storage-preflight.txt"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 250 > "$OUT/gpu.csv" 2>&1 &
monitor=$!
cleanup() { kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
printf 'start=%s\ncommand=flock -n /root/autodl-tmp/p2-3090.lock %s/autodl_storage_guard.sh run 268435456 -- env PATH=%s/venv-vllm-0.26.0/bin:$PATH %s %s/probe_awq_inprocess_decode.py --snapshot %s --output-dir %s/raw %s\n' "$(date -Is)" "$WORK" "$WORK" "$PY" "$HERE" "$SNAPSHOT" "$OUT" "${extra[*]:-}" > "$OUT/command.txt"
set +e
flock -n /root/autodl-tmp/p2-3090.lock "$WORK/autodl_storage_guard.sh" run 268435456 -- \
  env PATH="$WORK/venv-vllm-0.26.0/bin:$PATH" "$PY" "$HERE/probe_awq_inprocess_decode.py" \
  --snapshot "$SNAPSHOT" --output-dir "$OUT/raw" "${extra[@]}" > "$OUT/run.log" 2>&1
status=$?
set -e
cleanup
printf 'end=%s\nexit=%s\n' "$(date -Is)" "$status" >> "$OUT/command.txt"
printf '%s\n' "$status" > "$OUT/exit"
df -B1 / /root/autodl-tmp > "$OUT/df-after.txt"
nvidia-smi > "$OUT/gpu-after.txt"
find "$OUT" -type f ! -name sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$OUT/sha256.txt"
printf 'run_id=%s\nout=%s\nexit=%s\n' "$RUN_ID" "$OUT" "$status"
exit "$status"
