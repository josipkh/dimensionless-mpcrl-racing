#!/bin/bash
# Average the TF results of every formulation folder inside <results_folder>
# (requires https://github.com/janosh/tensorboard-reducer).
#
# For each immediate subfolder that holds per-seed runs (seed_*), tb-reducer
# writes the mean/std/min/max across the seeds into that same folder. If a
# reduction fails, its output is kept next to the folders it would have written
# (as <formulation>/AVERAGE_FAILED.log) and the script exits non-zero, so a
# failed averaging is easy to spot and to inspect later.
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <results_folder>"
    exit 1
fi
TOP_FOLDER="${1%/}"
if [ ! -d "$TOP_FOLDER" ]; then
    echo "Error: Folder '$TOP_FOLDER' does not exist."
    exit 1
fi

FAILURE_LOG="AVERAGE_FAILED.log"
processed=0
failed=0

for SUBFOLDER in $(ls "$TOP_FOLDER")
do
  FORM_FOLDER="$TOP_FOLDER/$SUBFOLDER"
  [ -d "$FORM_FOLDER" ] || continue
  # Only average folders that actually contain per-seed runs. This skips the
  # already-reduced folders (mean, std, min, max, ...) and any other folder.
  if ! compgen -G "$FORM_FOLDER/seed_*" > /dev/null; then
    continue
  fi

  processed=$((processed + 1))
  echo "processing $FORM_FOLDER"
  LOG_FILE="$FORM_FOLDER/$FAILURE_LOG"
  OUTPUT="$(tb-reducer "$FORM_FOLDER"/seed_* -o "$FORM_FOLDER"/ -r mean,std,min,max --overwrite --handle-dup-steps mean --lax-steps 2>&1)"
  RC=$?
  printf '%s\n' "$OUTPUT"

  if [ "$RC" -eq 0 ]; then
    rm -f "$LOG_FILE"
  else
    {
      echo "tb-reducer failed with exit code $RC"
      echo "date: $(date)"
      echo "command: tb-reducer $FORM_FOLDER/seed_* -o $FORM_FOLDER/ -r mean,std,min,max --overwrite --handle-dup-steps mean --lax-steps"
      echo "----------------------------------------"
      printf '%s\n' "$OUTPUT"
    } > "$LOG_FILE"
    echo "  ERROR: averaging $SUBFOLDER failed (exit $RC); wrote $LOG_FILE"
    failed=$((failed + 1))
  fi
  # seed_* (not *) averages only the actual per-seed runs, so re-running does not
  # fold the previously written mean/std/min/max folders back into the average
  # --handle-dup-steps resolves the situation with two values for the same timestep
  # (occurs when continuing the training with current settings)
  # --lax-steps tolerates the differing step ranges of the source/target transfer runs
done

if [ "$processed" -eq 0 ]; then
  echo "Error: no folder with per-seed runs (seed_*) found in '$TOP_FOLDER'."
  exit 1
fi

if [ "$failed" -ne 0 ]; then
  echo "Error: averaging failed for $failed folder(s). See the AVERAGE_FAILED.log files."
  exit 1
fi
