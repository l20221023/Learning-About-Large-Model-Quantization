#!/usr/bin/env bash
# Copy this directory to the existing AutoDL work tree; run only after dispatcher obtains GPU slot.
set -euo pipefail
WORK=/root/autodl-tmp/p2-deepseek-moe
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$WORK/autodl_storage_env.sh"
SNAPSHOT="$P2_MODELS_DIR/TechxGenus-DeepSeek-V2-Lite-Chat-AWQ-e9fdf80"
PY="$WORK/venv-vllm-0.26.0/bin/python"
VLLM="$WORK/venv-vllm-0.26.0/bin/vllm"
RUN_ID="Phase1-AWQ-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-3090"
OUT="$P2_EXPERIMENTS_DIR/$RUN_ID"
mkdir -p "$OUT"
[[ -x $PY && -x $VLLM && -d $SNAPSHOT ]] || { echo 'missing existing runtime or AWQ snapshot' >&2; exit 66; }
sha256sum "$HERE"/*.py "$HERE"/m0_4_seed_prompts.json > "$OUT/input-sha256.txt"
df -B1 / /root/autodl-tmp > "$OUT/df-before.txt"
nvidia-smi > "$OUT/gpu-before.txt"
"$WORK/autodl_storage_guard.sh" check 1073741824 > "$OUT/storage-preflight.txt"
cat > "$OUT/inner.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$WORK/venv-vllm-0.26.0/bin:$PATH"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 250 > "$OUT/gpu.csv" 2>&1 &
monitor=$!
"$VLLM" serve "$SNAPSHOT" --served-model-name p2-awq --host 127.0.0.1 --port 8765 \
  --max-model-len 256 --max-num-batched-tokens 256 --max-num-seqs 1 \
  --kv-cache-memory-bytes 536870912 --no-enable-prefix-caching --enforce-eager > "$OUT/server.log" 2>&1 &
server=$!
cleanup() { kill "$server" "$monitor" 2>/dev/null || true; wait "$server" "$monitor" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$server" 2>/dev/null; then echo 'server exited before health check' >&2; exit 1; fi
  if curl --fail --silent --max-time 2 http://127.0.0.1:8765/health >/dev/null; then ready=1; break; fi
  sleep 2
done
[[ $ready == 1 ]] || { echo 'server health timeout' >&2; exit 1; }
"$PY" "$HERE/collect_awq_baseline.py" --workload "$HERE/m0_4_seed_prompts.json" --output "$OUT/raw.jsonl" > "$OUT/collect.log" 2>&1
"$PY" "$HERE/analyze_awq_baseline.py" "$OUT/raw.jsonl" --output "$OUT/summary.json" > "$OUT/analyze.log" 2>&1
EOF
chmod +x "$OUT/inner.sh"
export WORK HERE OUT SNAPSHOT PY VLLM
printf 'start=%s\nrun_id=%s\ncommand=flock -n /root/autodl-tmp/p2-3090.lock %s/autodl_storage_guard.sh run 1073741824 -- bash %s/inner.sh\n' "$(date -Is)" "$RUN_ID" "$WORK" "$OUT" > "$OUT/command.txt"
set +e
flock -n /root/autodl-tmp/p2-3090.lock "$WORK/autodl_storage_guard.sh" run 1073741824 -- bash "$OUT/inner.sh" > "$OUT/runner.log" 2>&1
status=$?
set -e
printf 'end=%s\nexit=%s\n' "$(date -Is)" "$status" >> "$OUT/command.txt"
printf '%s\n' "$status" > "$OUT/exit"
df -B1 / /root/autodl-tmp > "$OUT/df-after.txt"
nvidia-smi > "$OUT/gpu-after.txt"
find "$OUT" -maxdepth 1 -type f ! -name sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$OUT/sha256.txt"
printf 'run_id=%s\nout=%s\nexit=%s\n' "$RUN_ID" "$OUT" "$status"
exit "$status"
