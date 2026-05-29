#!/bin/bash
# Run experiments per seed: default -> large sequentially, all seeds in parallel.
# Each seed chain runs in a single tmux session.

SCRIPT="PYTHONPATH=$(pwd) python race_car/scripts/run_sac_zop_transfer.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

declare -A FORMULATIONS=(
    ["dimensional"]=""
    ["dimensionless"]="--dimensionless"
)

declare -A TRAIN_STEPS=(
    ["default"]=50000
    ["large"]=100000
)

for FORMULATION in "${!FORMULATIONS[@]}"; do
    EXTRA_ARGS="${FORMULATIONS[$FORMULATION]}"

    for SEED in $(seq 0 4); do
        SESSION_NAME="${FORMULATION}_seed${SEED}_${TIMESTAMP}"

        CMD_DEFAULT="$SCRIPT --size default $EXTRA_ARGS --seed $SEED --timestamp $TIMESTAMP --train_steps ${TRAIN_STEPS[default]}"
        CMD_LARGE="$SCRIPT --size large $EXTRA_ARGS --seed $SEED --timestamp $TIMESTAMP --train_steps ${TRAIN_STEPS[large]}"

        CHAIN="bash -c '$CMD_DEFAULT && $CMD_LARGE; echo \"--- Done (exit $?) ---\"; tmux kill-session -t $SESSION_NAME'"

        echo "Starting tmux session: $SESSION_NAME"
        # echo "  Default: $CMD_DEFAULT"
        # echo "  Large:   $CMD_LARGE"
        tmux new-session -d -s "$SESSION_NAME" "$CHAIN"
    done

done

echo ""
echo "All sessions launched. Use 'tmux ls' to list them."
echo "Attach with: tmux attach -t <session_name>"