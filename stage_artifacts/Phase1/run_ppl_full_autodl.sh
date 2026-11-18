#!/usr/bin/env bash
set -uo pipefail
WORK=/root/autodl-tmp/p2-deepseek-moe
BASE=/root/autodl-tmp/experiments/Phase1-PPL-prep
OUT=/root/autodl-tmp/experiments/Phase1-PPL-full-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-3090
mkdir -p "$OUT"
df -B1 / /root/autodl-tmp > "$OUT/df-before.txt"
nvidia-smi > "$OUT/gpu-before.txt"
sha256sum "$BASE/wikitext_awq_ppl.py" "$BASE/test.parquet" > "$OUT/inputs-sha256.txt"
"$WORK/autodl_storage_guard.sh" check 2147483648 > "$OUT/preflight.txt" || exit $?
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$WORK/venv-vllm-0.26.0/bin:$PATH"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu.csv" 2>&1 &
monitor=$!
flock -n /root/autodl-tmp/p2-3090.lock "$WORK/autodl_storage_guard.sh" run 2147483648 -- \
  "$WORK/venv-vllm-0.26.0/bin/python" "$BASE/wikitext_awq_ppl.py" \
  --snapshot "$WORK/models/TechxGenus-DeepSeek-V2-Lite-Chat-AWQ-e9fdf80" \
  --data-parquet "$BASE/test.parquet" --output-dir "$OUT/result" \
  > "$OUT/runner.log" 2>&1
status=$?
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
echo "$status" > "$OUT/exit-code.txt"
df -B1 / /root/autodl-tmp > "$OUT/df-after.txt"
nvidia-smi > "$OUT/gpu-after.txt"
sha256sum "$OUT"/result/* "$OUT"/gpu.csv > "$OUT/output-sha256.txt" 2>/dev/null || true
exit "$status"
