#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export P2_SRC_PATH=/root/autodl-tmp/p2-deepseek-moe/src
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 250 > "$OUT/gpu0.csv" 2>&1 &
sampler=$!
cleanup() { kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
set +e
timeout 5400 "$PY" "$BASE/trace_bf16_decode.py" --snapshot "$SNAPSHOT" \
  --workload "$BASE/workloads/m0_4_seed_prompts.json" \
  --output-dir "$OUT/trace" --decode-tokens 32 > "$OUT/trace.log" 2>&1
status=$?
set -e
echo "$status" > "$OUT/exit"
cleanup
trap - EXIT INT TERM
exit "$status"
