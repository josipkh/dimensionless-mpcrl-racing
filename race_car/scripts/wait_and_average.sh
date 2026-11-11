#!/bin/bash
# Wait for a batch of tmux training sessions to finish, then average their
# TensorBoard logs with average_results.sh.
#
# The run_all_* scripts launch one tmux session per (size x formulation x seed),
# all named with the same <timestamp>. This watcher polls tmux until no such
# session is left and then averages the results.
#
# Averaging is always done separately for the seeds of each formulation
# ("dimensional" and "dimensionless"), and for every timestamp folder found
# under <results_folder>. This is delegated to average_results.sh, which
# averages every formulation subfolder of the folder it is given, so it is
# called once per timestamp folder. Passing a date folder (which holds several
# timestamps) therefore averages each timestamp separately instead of mixing
# its formulations.
#
# It is normally started automatically by the run_all_* scripts; it can also be
# run by hand for a batch that is already in flight.
#
# Usage:
#   wait_and_average.sh <timestamp> <results_folder> [poll_seconds] [self_session]
#
# Arguments:
#   timestamp       Batch timestamp, as used in the tmux session names and in
#                   the output folder name (e.g. 20261111_124035).
#   results_folder  Folder to average. Either a single timestamp folder
#                   (output/2026_11_11/12_40_35_sac_zop_transfer) or a date
#                   folder holding several timestamps (output/2026_11_11).
#   poll_seconds    How often to check tmux (default: 60).
#   self_session    Optional tmux session name of this watcher. If given and
#                   averaging succeeds, the watcher kills its own session. On
#                   failure the session is left open so the error can be read.
#
# Examples:
#   wait_and_average.sh 20261111_124035 output/2026_11_11/12_40_35_sac_zop_transfer
#   wait_and_average.sh 20261111_124035 output/2026_11_11 30 averager_20261111_124035

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

# Timestamp folders are the parents of the formulation folders, which are the
# parents of the seed_* folders. De-duplicated, so a folder holding several
# timestamps yields each of them once and a single timestamp folder yields
# itself. average_results.sh then averages dimensional / dimensionless
# separately for each of them.
timestamp_folders() {
    find "$RESULTS_FOLDER" -maxdepth 3 -type d -name 'seed_*' -printf '%h\n' 2>/dev/null \
        | sort -u \
        | while IFS= read -r formulation; do dirname "$formulation"; done \
        | sort -u
}

echo "[wait_and_average] waiting for sessions with timestamp $TIMESTAMP to finish (poll ${POLL_SECONDS}s)"
while sessions_running | grep -q .; do
    sleep "$POLL_SECONDS"
done

if [ ! -d "$RESULTS_FOLDER" ]; then
    echo "[wait_and_average] results folder '$RESULTS_FOLDER' not found; nothing to average."
    exit 1
fi

mapfile -t FOLDERS < <(timestamp_folders)
if [ "${#FOLDERS[@]}" -eq 0 ]; then
    echo "[wait_and_average] no seed_* folders found under '$RESULTS_FOLDER'; nothing to average."
    exit 1
fi

echo "[wait_and_average] all sessions finished; averaging ${#FOLDERS[@]} timestamp folder(s)"
rc=0
for folder in "${FOLDERS[@]}"; do
    echo "[wait_and_average] averaging $folder (dimensional / dimensionless separately)"
    "$AVERAGE_SCRIPT" "$folder" || rc=$?
done

if [ "$rc" -eq 0 ]; then
    echo "[wait_and_average] averaging done."
    if [ -n "$SELF_SESSION" ]; then
        tmux kill-session -t "$SELF_SESSION" 2>/dev/null || true
    fi
    exit 0
else
    echo "[wait_and_average] averaging FAILED (exit $rc); leaving this session open."
    exit "$rc"
fi
