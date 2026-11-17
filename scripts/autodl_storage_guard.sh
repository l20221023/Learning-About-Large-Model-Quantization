#!/usr/bin/env bash
# Capacity preflight and monitored execution for AutoDL data/system volumes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/autodl_storage_env.sh"
SYSTEM_RESERVE=$((8 * 1024 * 1024 * 1024))
DATA_RESERVE=$((10 * 1024 * 1024 * 1024))

free_bytes() { df -P -B1 "$1" | awk 'NR==2 {print $4}'; }

check_capacity() {
    local data_peak=${1:?peak data bytes required}
    local system_peak=${2:-0}
    local system_free data_free
    [[ $data_peak =~ ^[0-9]+$ && $system_peak =~ ^[0-9]+$ ]] || { echo 'peak byte counts must be nonnegative integers' >&2; return 64; }
    system_free=$(free_bytes /)
    data_free=$(free_bytes "$AUTODL_DATA_ROOT")
    printf 'system_free=%s system_peak=%s system_reserve=%s\n' "$system_free" "$system_peak" "$SYSTEM_RESERVE"
    printf 'data_free=%s data_peak=%s data_reserve=%s\n' "$data_free" "$data_peak" "$DATA_RESERVE"
    (( system_free - system_peak >= SYSTEM_RESERVE )) || { echo 'BLOCKED: system volume would breach 8 GiB reserve' >&2; return 75; }
    (( data_free - data_peak >= DATA_RESERVE )) || { echo 'BLOCKED: data volume would breach 10 GiB reserve' >&2; return 75; }
}

if [[ ${1:-} == check ]]; then
    shift
    check_capacity "${1:-}" "${2:-0}"
    exit $?
fi
if [[ ${1:-} != run || ${2:-} == '' || ${3:-} != -- ]]; then
    echo "usage: $0 check PEAK_DATA_BYTES [PEAK_SYSTEM_BYTES] | run PEAK_DATA_BYTES -- COMMAND [ARGS...]" >&2
    exit 64
fi
peak=$2
shift 3
check_capacity "$peak" 0
stop_marker=$(mktemp "$P2_TMP_DIR/storage-guard-stop.XXXXXX")
rm -f "$stop_marker"
setsid -- "$@" &
target=$!
monitor_space() {
    while kill -0 "$target" 2>/dev/null; do
        sleep 5
        system_free=$(free_bytes /)
        data_free=$(free_bytes "$AUTODL_DATA_ROOT")
        if (( system_free < SYSTEM_RESERVE || data_free < DATA_RESERVE )); then
            printf 'SPACE_GUARD_STOP time=%s system_free=%s data_free=%s\n' "$(date -Is)" "$system_free" "$data_free" >&2
            : > "$stop_marker"
            kill -INT -- "-$target" 2>/dev/null || true
            for _ in $(seq 1 60); do
                kill -0 "$target" 2>/dev/null || return
                sleep 1
            done
            kill -TERM -- "-$target" 2>/dev/null || true
            return
        fi
    done
}
monitor_space &
monitor=$!
set +e
wait "$target"
status=$?
set -e
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
if [[ -f $stop_marker ]]; then
    rm -f "$stop_marker"
    echo 'SPACE_GUARD_STOP: task interrupted at configured reserve; inspect partial files before resuming.' >&2
    exit 75
fi
exit "$status"
