#!/usr/bin/env bash
set -euo pipefail
WORK=/root/autodl-tmp/p2-deepseek-moe
source "$WORK/autodl_storage_env.sh"
RUN_ID=${1:?usage: package_m0_4_autodl_evidence.sh M0-4-run-id}
RUN="$P2_EXPERIMENTS_DIR/$RUN_ID"
LOG="$P2_LOGS_DIR/$RUN_ID"
ARCHIVE_DIR="$P2_LOGS_DIR/evidence"
ARCHIVE="$ARCHIVE_DIR/$RUN_ID.tar.gz"
test -f "$RUN/trace/manifest.json" && test -f "$RUN/trace/router_trace.jsonl"
test "$(cat "$LOG/exit")" = 0
mkdir -p "$ARCHIVE_DIR"
test ! -e "$ARCHIVE" || { echo "archive already exists: $ARCHIVE" >&2; exit 73; }
"$WORK/autodl_storage_guard.sh" check 268435456
guard_log="$P2_LOGS_DIR/storage-guard-capacity-test.log"
if "$WORK/autodl_storage_guard.sh" run 999999999999 -- touch "$P2_TMP_DIR/storage-guard-should-not-run-$RUN_ID" >"$guard_log" 2>&1; then
    echo 'capacity guard unexpectedly allowed an oversized task' >&2
    exit 70
else
    guard_status=$?
fi
test "$guard_status" = 75
test ! -e "$P2_TMP_DIR/storage-guard-should-not-run-$RUN_ID"
echo 'insufficient_space_preflight=PASS; run_command_not_started=PASS' >> "$guard_log"
bash "$WORK/audit_autodl_storage.sh" > "$P2_LOGS_DIR/storage-audit-20261117.log" 2>&1
tar -czf "$ARCHIVE" \
    -C "$P2_EXPERIMENTS_DIR" "$RUN_ID" \
    -C "$P2_LOGS_DIR" "$RUN_ID" storage-config-verification.log storage-audit-20261117.log storage-guard-capacity-test.log \
    -C "$P2_WORK_ROOT/results" M0-4-20261117T230000-3090 \
    -C "$P2_LOGS_DIR" M0-4-20261117T231610-3090 M0-4-20261117T231827-3090 \
        M0-4-20261117T232117-3090 M0-4-20261117T232556-3090 \
        M0-4-20261117T233144-3090 M0-4-20261117T233421-3090
sha256sum "$ARCHIVE"
du -h "$ARCHIVE"
printf 'run_id=%s\narchive=%s\n' "$RUN_ID" "$ARCHIVE"
