#!/usr/bin/env bash
# Console-log helper for project_commands.sh.
# This file is sourced; it is not intended to be executed directly.

start_run_log() {
    local project_root="$1"
    local action="${2:-all}"
    local safe_action="${action//[^[:alnum:]_.-]/_}"
    local started_at
    local session_id

    started_at="$(date +%s)"
    session_id="$(date +%Y%m%d_%H%M%S)_${safe_action}_$$"

    export RUN_LOG_DIR="${RUN_LOG_DIR:-$project_root/results/session_logs}"
    export RUN_LOG_FILE="$RUN_LOG_DIR/${session_id}.log"
    export RUN_LOG_STARTED_AT="$started_at"

    mkdir -p "$RUN_LOG_DIR"
    ln -sfn "$(basename "$RUN_LOG_FILE")" "$RUN_LOG_DIR/latest_${safe_action}.log"

    exec > >(tee -a "$RUN_LOG_FILE") 2>&1

    printf '%s\n' "================================================================================"
    printf 'session    : %s\n' "$session_id"
    printf 'action     : %s\n' "$action"
    printf 'started    : %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'host       : %s\n' "$(hostname)"
    printf 'pid        : %s\n' "$$"
    printf 'root       : %s\n' "$project_root"
    printf 'devices    : %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'console log: %s\n' "$RUN_LOG_FILE"
    printf '%s\n' "================================================================================"

    trap '_finish_run_log "$?"' EXIT
}

_finish_run_log() {
    local exit_code="$1"
    local finished_at
    local elapsed

    finished_at="$(date +%s)"
    elapsed=$((finished_at - RUN_LOG_STARTED_AT))

    printf '%s\n' "================================================================================"
    printf 'finished   : %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'exit code  : %s\n' "$exit_code"
    printf 'elapsed    : %02d:%02d:%02d\n' \
        "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
    printf '%s\n' "================================================================================"
}
