#!/bin/bash
# Run all 4 combinations of size x dimensionless x 5 seeds in separate tmux sessions
# Sessions can be listed with: tmux ls
# Attach to a session with: tmux attach -t <session_name>
SCRIPT="PYTHONPATH=$(pwd) python race_car/scripts/run_sac_zop.py"
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
echo ""
echo "All sessions launched. Use 'tmux ls' to list them."
echo "Attach with: tmux attach -t <session_name>"