#!/bin/bash
# Run all 4 combinations of size x dimensionless in separate tmux sessions
# Sessions can be listed with: tmux ls
# Attach to a session with: tmux attach -t <session_name>
SCRIPT="PYTHONPATH=$(pwd) python race_car/run_controller.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
declare -A RUNS=(
    ["default_dimensional"]="--size default"
    ["default_dimensionless"]="--size default --dimensionless"
    ["large_dimensional"]="--size large"
    ["large_dimensionless"]="--size large --dimensionless"
)
for SESSION in "${!RUNS[@]}"; do
    ARGS="${RUNS[$SESSION]}"
    CMD="$SCRIPT $ARGS --timestamp $TIMESTAMP"
    SESSION_NAME="${SESSION}_${TIMESTAMP}"
    echo "Starting tmux session: $SESSION_NAME"
    echo "  Command: $CMD"
    tmux new-session -d -s "$SESSION_NAME" "bash -c '$CMD; echo \"--- Done (exit $?) ---\"; tmux kill-session -t $SESSION_NAME'"
done
echo ""
echo "All sessions launched. Use 'tmux ls' to list them."
echo "Attach with: tmux attach -t <session_name>"