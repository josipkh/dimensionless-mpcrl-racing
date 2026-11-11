#!/bin/bash
# Run all 4 combinations of size x dimensionless x 5 seeds in separate tmux sessions
# Sessions can be listed with: tmux ls
# Attach to a session with: tmux attach -t <session_name>
#
# When every session of the batch has finished, a watcher session averages the
# per-seed TensorBoard logs (race_car/scripts/average_results.sh). Set
# AUTO_AVERAGE=0 to skip this.
SCRIPT="PYTHONPATH=$(pwd) python race_car/scripts/run_sac_zop.py"
WATCHER_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wait_and_average.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
declare -A RUNS=(
    ["default_dimensional"]="--size default"
    ["default_dimensionless"]="--size default --dimensionless"
    ["large_dimensional"]="--size large"
    ["large_dimensionless"]="--size large --dimensionless"
)
for SESSION in "${!RUNS[@]}"; do
    ARGS="${RUNS[$SESSION]}"
    for SEED in {0..4}; do
        CMD="$SCRIPT $ARGS --seed $SEED --timestamp $TIMESTAMP"
        SESSION_NAME="${SESSION}_seed${SEED}_${TIMESTAMP}"
        echo "Starting tmux session: $SESSION_NAME"
        echo "  Command: $CMD"
        tmux new-session -d -s "$SESSION_NAME" "bash -c '$CMD; echo \"--- Done (exit $?) ---\"; tmux kill-session -t $SESSION_NAME'"
    done
done

# Average the per-seed logs once the whole batch has finished. The output path
# mirrors the one built by run_sac_zop.py from the timestamp.
if [ "${AUTO_AVERAGE:-1}" = "1" ]; then
    OUTPUT_FOLDER="output/${TIMESTAMP:0:4}_${TIMESTAMP:4:2}_${TIMESTAMP:6:2}/${TIMESTAMP:9:2}_${TIMESTAMP:11:2}_${TIMESTAMP:13:2}_sac_zop"
    WATCHER_SESSION="averager_${TIMESTAMP}"
    echo ""
    echo "Starting watcher session: $WATCHER_SESSION"
    echo "  Will average $OUTPUT_FOLDER once all sessions finish."
    tmux new-session -d -s "$WATCHER_SESSION" "bash '$WATCHER_SCRIPT' $TIMESTAMP '$OUTPUT_FOLDER' 60 '$WATCHER_SESSION'"
fi

echo ""
echo "All sessions launched. Use 'tmux ls' to list them."
echo "Attach with: tmux attach -t <session_name>"
