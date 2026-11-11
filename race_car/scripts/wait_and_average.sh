#!/bin/bash
# Wait for a batch of tmux training sessions to finish, then average their
# TensorBoard logs with average_results.sh.
#
# The run_all_* scripts launch one tmux session per (size x formulation x seed),
# all named with the same <timestamp>. This watcher polls tmux until no such
# session is left and then averages the results folder. It is normally started
# automatically by the run_all_* scripts; it can also be run by hand for a batch
# that is already in flight.
#
# Usage:
#   wait_and_average.sh <timestamp> <results_folder> [poll_seconds] [self_session]
#
# Arguments:
#   timestamp       Batch timestamp, as used in the tmux session names and in
#                   the output folder name (e.g. 20261111_124035).
#   results_folder  Folder passed to average_results.sh, e.g.
#                   output/2026_11_11/12_40_35_sac_zop_transfer
#   poll_seconds    How often to check tmux (default: 60).
#   self_session    Optional tmux session name of this watcher. If given and
#                   averaging succeeds, the watcher kills its own session. On
#                   failure the session is left open so the error can be read.
#
# Examples:
#   wait_and_average.sh 20261111_124035 output/2026_11_11/12_40_35_sac_zop_transfer
#   wait_and_average.sh 20261111_124035 output/2026_11_11/12_40_35_sac_zop_transfer 30 averager_20261111_124035

set -u

TIMESTAMP="${1:-}"
RESULTS_FOLDER="${2:-}"
RESULTS_FOLDER="${RESULTS_FOLDER%/}"
POLL_SECONDS="${3:-60}"
SELF_SESSION="${4:-}"

if [ -z "$TIMESTAMP" ] || [ -z "$RESULTS_FOLDER" ]; then
    echo "Usage: $0 <timestamp> <results_folder> [poll_seconds] [self_session]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVERAGE_SCRIPT="$SCRIPT_DIR/average_results.sh"

# Training sessions are named "<...>_<TIMESTAMP>". This watcher is named
# "averager_<TIMESTAMP>", so exclude it from the running-session check.
sessions_running() {
    tmux ls 2>/dev/null | grep -F "_${TIMESTAMP}" | grep -v '^averager_'
}

echo "[wait_and_average] waiting for sessions with timestamp $TIMESTAMP to finish (poll ${POLL_SECONDS}s)"
while sessions_running | grep -q .; do
    sleep "$POLL_SECONDS"
done

if [ ! -d "$RESULTS_FOLDER" ]; then
    echo "[wait_and_average] results folder '$RESULTS_FOLDER' not found; nothing to average."
    exit 1
fi

echo "[wait_and_average] all sessions finished; averaging $RESULTS_FOLDER"
if "$AVERAGE_SCRIPT" "$RESULTS_FOLDER"; then
    echo "[wait_and_average] averaging done."
    if [ -n "$SELF_SESSION" ]; then
        tmux kill-session -t "$SELF_SESSION" 2>/dev/null || true
    fi
    exit 0
else
    rc=$?
    echo "[wait_and_average] averaging FAILED (exit $rc); leaving this session open."
    exit "$rc"
fi
