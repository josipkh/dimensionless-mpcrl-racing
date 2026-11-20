#!/bin/bash
# Run transfer experiments per seed: source car -> target car sequentially, all seeds in parallel.
# Each seed chain runs in a single tmux session.
#
# The transfer direction is selected with TRANSFER_DIRECTION (default: both):
#   both              run both directions in one batch folder (default)
#   default_to_large  small -> large  (pretrain the default car, fine-tune the large car)
#   large_to_default  large -> small  (pretrain the large car, fine-tune the default car)
#
# Both runs of a chain share the same output directory, so the target run continues
# from the source checkpoint. The source car is pretrained briefly, the target car
# is fine-tuned longer. The direction is recorded in the output folder name and in
# a transfer.json manifest next to the results, so no manual bookkeeping is needed.
# The default batch folder is direction-neutral and contains one subfolder per
# direction, plus a manifest describing both of them.
#
# When every session of the batch has finished, a watcher session averages the
# per-seed TensorBoard logs (race_car/scripts/average_results.sh). Set
# AUTO_AVERAGE=0 to skip this.
#
# Usage:
#   ./race_car/scripts/run_all_sac_zop_transfer.sh                              # both directions
#   TRANSFER_DIRECTION=default_to_large ./race_car/scripts/run_all_sac_zop_transfer.sh
#   TRANSFER_DIRECTION=large_to_default ./race_car/scripts/run_all_sac_zop_transfer.sh
#   AUTO_AVERAGE=0 ./race_car/scripts/run_all_sac_zop_transfer.sh

TRANSFER_DIRECTION="${TRANSFER_DIRECTION:-both}"

SCRIPT="PYTHONPATH=$(pwd) python race_car/scripts/run_sac_zop_transfer.py"
WATCHER_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wait_and_average.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

declare -A FORMULATIONS=(
    ["dimensional"]=""
    ["dimensionless"]="--dimensionless"
)

# Step budget: pretrain the source car briefly, then fine-tune the target car longer.
SOURCE_STEPS=50000
TARGET_STEPS=100000

declare -A SOURCE_CAR=(
    ["default_to_large"]="default"
    ["large_to_default"]="large"
)
declare -A TARGET_CAR=(
    ["default_to_large"]="large"
    ["large_to_default"]="default"
)

case "$TRANSFER_DIRECTION" in
    default_to_large|large_to_default)
        DIRECTIONS=("$TRANSFER_DIRECTION")
        BATCH_SUFFIX="sac_zop_transfer_${TRANSFER_DIRECTION}"
        ;;
    both)
        DIRECTIONS=("default_to_large" "large_to_default")
        BATCH_SUFFIX="sac_zop_transfer"
        ;;
    *)
        echo "Unknown TRANSFER_DIRECTION '$TRANSFER_DIRECTION' (use 'default_to_large', 'large_to_default' or 'both')"
        exit 1
        ;;
esac

# One folder per batch. Explicit single-direction runs keep the old flat layout;
# the default batch uses a direction-neutral root with one subfolder per direction.
BATCH_ROOT="output/${TIMESTAMP:0:4}_${TIMESTAMP:4:2}_${TIMESTAMP:6:2}/${TIMESTAMP:9:2}_${TIMESTAMP:11:2}_${TIMESTAMP:13:2}_${BATCH_SUFFIX}"
mkdir -p "$BATCH_ROOT"

# Write all manifests before training starts, so the tmux sessions never race.
# Per-direction manifests keep the original shape; both-direction batches
# additionally get a root manifest listing both directions.
DIRECTION_SPECS=()
for DIRECTION in "${DIRECTIONS[@]}"; do
    DIRECTION_SPECS+=("$DIRECTION:${SOURCE_CAR[$DIRECTION]}:${TARGET_CAR[$DIRECTION]}")
done
python - "$BATCH_ROOT" "$SOURCE_STEPS" "$TARGET_STEPS" "$TIMESTAMP" "${DIRECTION_SPECS[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root, source_steps, target_steps, timestamp, *specs = sys.argv[1:]
common = {
    "formulations": ["dimensional", "dimensionless"],
    "seeds": list(range(5)),
    "timestamp": timestamp,
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip(),
}


def entry(direction, source, target):
    return {
        "direction": direction,
        "label": "small -> large" if direction == "default_to_large" else "large -> small",
        "source": source,
        "target": target,
        "source_steps": int(source_steps),
        "target_steps": int(target_steps),
    }


entries = {}
for spec in specs:
    direction, source, target = spec.split(":")
    entries[direction] = entry(direction, source, target)

root = Path(root)
if len(entries) > 1:
    (root / "transfer.json").write_text(
        json.dumps(dict(common, directions=entries), indent=2) + "\n"
    )

for direction, info in entries.items():
    folder = root if len(entries) == 1 else root / direction
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transfer.json").write_text(json.dumps(dict(common, **info), indent=2) + "\n")
PY

for DIRECTION in "${DIRECTIONS[@]}"; do
    SOURCE="${SOURCE_CAR[$DIRECTION]}"
    TARGET="${TARGET_CAR[$DIRECTION]}"
    if [ "${#DIRECTIONS[@]}" -gt 1 ]; then
        RUN_ROOT="$BATCH_ROOT/$DIRECTION"
    else
        RUN_ROOT="$BATCH_ROOT"
    fi

    for FORMULATION in "${!FORMULATIONS[@]}"; do
        EXTRA_ARGS="${FORMULATIONS[$FORMULATION]}"

        for SEED in $(seq 0 4); do
            SESSION_NAME="${FORMULATION}_${DIRECTION}_seed${SEED}_${TIMESTAMP}"

            CMD_SOURCE="$SCRIPT --size $SOURCE --direction $DIRECTION --phase source $EXTRA_ARGS --seed $SEED --timestamp $TIMESTAMP --run_root $RUN_ROOT --train_steps $SOURCE_STEPS"
            CMD_TARGET="$SCRIPT --size $TARGET --direction $DIRECTION --phase target $EXTRA_ARGS --seed $SEED --timestamp $TIMESTAMP --run_root $RUN_ROOT --train_steps $TARGET_STEPS"

            CHAIN="bash -c '$CMD_SOURCE && $CMD_TARGET; echo \"--- Done (exit $?) ---\"; tmux kill-session -t $SESSION_NAME'"

            echo "Starting tmux session: $SESSION_NAME"
            tmux new-session -d -s "$SESSION_NAME" "$CHAIN"
        done
    done
done

# Average the per-seed logs once the whole batch has finished. The output path
# mirrors the one built by run_sac_zop_transfer.py from the timestamp.
if [ "${AUTO_AVERAGE:-1}" = "1" ]; then
    OUTPUT_FOLDER="$BATCH_ROOT"
    WATCHER_SESSION="averager_${TIMESTAMP}"
    echo ""
    echo "Starting watcher session: $WATCHER_SESSION"
    echo "  Will average $OUTPUT_FOLDER once all sessions finish."
    tmux new-session -d -s "$WATCHER_SESSION" "bash '$WATCHER_SCRIPT' $TIMESTAMP '$OUTPUT_FOLDER' 60 '$WATCHER_SESSION'"
fi

echo ""
echo "All sessions launched. Use 'tmux ls' to list them."
echo "Attach with: tmux attach -t <session_name>"
