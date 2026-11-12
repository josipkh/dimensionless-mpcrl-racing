#!/usr/bin/env bash
# Run the long RL experiments.  Unlike the original tmux launchers, this works
# in a container and limits the number of concurrent jobs.
#
# Suites:
#   main      SAC-ZOP, 4 size x formulation combinations, all seeds
#   transfer  default -> large transfer chain per formulation and seed
#   baseline  tracking-controller baseline, 4 combinations
#   all       baseline + main + transfer
set -Eeuo pipefail

ROOT="${REPO_ROOT:-/workspace}"
cd "$ROOT"

SUITE="${1:-main}"
case "$SUITE" in
    main|transfer|baseline|all) ;;
    *)
        echo "Unknown RL suite: $SUITE" >&2
        echo "Available suites: main, transfer, baseline, all" >&2
        exit 2
        ;;
esac

STAMP="${REPRO_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
JOBS="${RL_JOBS:-$(nproc)}"
MAIN_STEPS="${RL_MAIN_STEPS:-100000}"
TRANSFER_DEFAULT_STEPS="${RL_TRANSFER_DEFAULT_STEPS:-50000}"
TRANSFER_LARGE_STEPS="${RL_TRANSFER_LARGE_STEPS:-100000}"

read -r -a SEEDS <<< "${RL_SEEDS:-0 1 2 3 4}"
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "RL_SEEDS must contain at least one seed." >&2
    exit 2
fi

LOG_DIR="$ROOT/output/docker/logs/$STAMP"
mkdir -p "$LOG_DIR"
MANIFEST="$LOG_DIR/manifest.txt"

{
    echo "suite: $SUITE"
    echo "run_id: $STAMP"
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit: ${GIT_COMMIT:-unknown}"
    echo "git_dirty: ${GIT_DIRTY:-unknown}"
    echo "seeds: ${SEEDS[*]}"
    echo "jobs: $JOBS"
    echo "main_train_steps: $MAIN_STEPS"
    echo "transfer_default_train_steps: $TRANSFER_DEFAULT_STEPS"
    echo "transfer_large_train_steps: $TRANSFER_LARGE_STEPS"
    echo
    echo "--- Commands ---"
} > "$MANIFEST"

SPECS="$(mktemp)"
trap 'rm -f "$SPECS"' EXIT

append_spec() {
    printf '%s\t%s\n' "$1" "$2" >> "$SPECS"
}

add_baseline() {
    local size formulation flag name
    for size in default large; do
        for formulation in dimensional dimensionless; do
            if [[ "$formulation" == "dimensionless" ]]; then
                flag="--dimensionless"
            else
                flag=""
            fi
            name="baseline_${size}_${formulation}"
            append_spec "$name" "python race_car/scripts/run_controller.py --size $size $flag --timestamp $STAMP"
        done
    done
}

add_main() {
    local size formulation flag seed name
    for size in default large; do
        for formulation in dimensional dimensionless; do
            if [[ "$formulation" == "dimensionless" ]]; then
                flag="--dimensionless"
            else
                flag=""
            fi
            for seed in "${SEEDS[@]}"; do
                name="sac_zop_${size}_${formulation}_seed${seed}"
                append_spec "$name" "python race_car/scripts/run_sac_zop.py --size $size $flag --seed $seed --timestamp $STAMP --train_steps $MAIN_STEPS"
            done
        done
    done
}

add_transfer() {
    local formulation flag seed name default_cmd large_cmd
    for formulation in dimensional dimensionless; do
        if [[ "$formulation" == "dimensionless" ]]; then
            flag="--dimensionless"
        else
            flag=""
        fi
        for seed in "${SEEDS[@]}"; do
            name="sac_zop_transfer_${formulation}_seed${seed}"
            default_cmd="python race_car/scripts/run_sac_zop_transfer.py --size default $flag --seed $seed --timestamp $STAMP --train_steps $TRANSFER_DEFAULT_STEPS"
            large_cmd="python race_car/scripts/run_sac_zop_transfer.py --size large $flag --seed $seed --timestamp $STAMP --train_steps $TRANSFER_LARGE_STEPS"
            # The two runs share an output directory; the second run resumes
            # from the checkpoint written by the first one.
            append_spec "$name" "$default_cmd && $large_cmd"
        done
    done
}

case "$SUITE" in
    baseline) add_baseline ;;
    main) add_main ;;
    transfer) add_transfer ;;
    all)
        add_baseline
        add_main
        add_transfer
        ;;
esac

cat "$SPECS" | cut -f2- >> "$MANIFEST"

JOB_COUNT="$(wc -l < "$SPECS" | tr -d ' ')"
echo "RL suite: $SUITE"
echo "Run id:   $STAMP"
echo "Jobs:     $JOB_COUNT (parallelism: $JOBS)"
echo "Logs:     $LOG_DIR"
echo

run_spec() {
    local spec="$1"
    local name cmd logfile status
    IFS=$'\t' read -r name cmd <<< "$spec"
    logfile="$LOG_DIR/$name.log"

    printf '[%s] START %s\n' "$(date -u +%H:%M:%SZ)" "$name"
    if ( cd "$ROOT" && eval "$cmd" ) > "$logfile" 2>&1; then
        printf '[%s] DONE  %s\n' "$(date -u +%H:%M:%SZ)" "$name"
    else
        status=$?
        printf '[%s] FAIL  %s (exit %s; see %s)\n' "$(date -u +%H:%M:%SZ)" "$name" "$status" "$logfile"
        return "$status"
    fi
}
export -f run_spec
export ROOT LOG_DIR

if ! xargs -P "$JOBS" -d '\n' -I '{}' bash -c 'run_spec "$1"' _ '{}' < "$SPECS"; then
    echo
    echo "At least one RL run failed. Check the logs in: $LOG_DIR" >&2
    exit 1
fi

echo
echo "RL suite complete: $SUITE"
echo "Logs: $LOG_DIR"
echo "Experiment output: $ROOT/output"
