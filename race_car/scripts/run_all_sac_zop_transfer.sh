#!/bin/bash
# Run transfer experiments per seed: source car -> target car sequentially, all seeds in parallel.
# Each seed chain runs in a single tmux session.
#
# The transfer direction is selected with TRANSFER_DIRECTION:
#   default_to_large  small -> large  (pretrain the default car, fine-tune the large car)
#   large_to_default  large -> small  (pretrain the large car, fine-tune the default car)
#
# Both runs of a chain share the same output directory, so the target run continues
# from the source checkpoint. The source car is pretrained briefly, the target car
# is fine-tuned longer.
#
# Usage:
#   ./race_car/scripts/run_all_sac_zop_transfer.sh
#   TRANSFER_DIRECTION=large_to_default ./race_car/scripts/run_all_sac_zop_transfer.sh

TRANSFER_DIRECTION="${TRANSFER_DIRECTION:-default_to_large}"

SCRIPT="PYTHONPATH=$(pwd) python race_car/scripts/run_sac_zop_transfer.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

declare -A FORMULATIONS=(
    ["dimensional"]=""
    ["dimensionless"]="--dimensionless"
)

# Step budget: pretrain the source car briefly, then fine-tune the target car longer.
SOURCE_STEPS=50000
TARGET_STEPS=100000

case "$TRANSFER_DIRECTION" in
    default_to_large) SOURCE="default"; TARGET="large" ;;
    large_to_default) SOURCE="large";   TARGET="default" ;;
    *)
        echo "Unknown TRANSFER_DIRECTION '$TRANSFER_DIRECTION' (use 'default_to_large' or 'large_to_default')"
        exit 1
        ;;
esac

for FORMULATION in "${!FORMULATIONS[@]}"; do
    EXTRA_ARGS="${FORMULATIONS[$FORMULATION]}"

    for SEED in $(seq 0 4); do
        SESSION_NAME="${FORMULATION}_${TRANSFER_DIRECTION}_seed${SEED}_${TIMESTAMP}"

        CMD_SOURCE="$SCRIPT --size $SOURCE $EXTRA_ARGS --seed $SEED --timestamp $TIMESTAMP --train_steps $SOURCE_STEPS"
        CMD_TARGET="$SCRIPT --size $TARGET $EXTRA_ARGS --seed $SEED --timestamp $TIMESTAMP --train_steps $TARGET_STEPS"

        CHAIN="bash -c '$CMD_SOURCE && $CMD_TARGET; echo \"--- Done (exit $?) ---\"; tmux kill-session -t $SESSION_NAME'"

        echo "Starting tmux session: $SESSION_NAME"
        # echo "  Source ($SOURCE): $CMD_SOURCE"
        # echo "  Target ($TARGET): $CMD_TARGET"
        tmux new-session -d -s "$SESSION_NAME" "$CHAIN"
    done

done

echo ""
echo "All sessions launched. Use 'tmux ls' to list them."
echo "Attach with: tmux attach -t <session_name>"
