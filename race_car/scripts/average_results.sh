#!/bin/bash
# Average the TF results across all 4 combinations of size x dimensionless
# (requires https://github.com/janosh/tensorboard-reducer)
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <results_folder>"
    exit 1
fi
TOP_FOLDER="${1%/}"
if [ ! -d "$TOP_FOLDER" ]; then
    echo "Error: Folder '$TOP_FOLDER' does not exist."
    exit 1
fi
for SUBFOLDER in $(ls "$TOP_FOLDER")
do
  echo "processing $TOP_FOLDER/$SUBFOLDER"
  tb-reducer "$TOP_FOLDER/$SUBFOLDER"/seed_* -o "$TOP_FOLDER/$SUBFOLDER"/ -r mean,std,min,max --overwrite --handle-dup-steps mean --lax-steps
  # seed_* (not *) averages only the actual per-seed runs, so re-running does not
  # fold the previously written mean/std/min/max folders back into the average
  # --handle-dup-steps resolves the situation with two values for the same timestep
  # (occurs when continuing the training with current settings)
  # --lax-steps tolerates the differing step ranges of the source/target transfer runs
done