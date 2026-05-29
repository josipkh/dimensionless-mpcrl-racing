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
  tb-reducer "$TOP_FOLDER/$SUBFOLDER"/* -o "$TOP_FOLDER/$SUBFOLDER"/ -r mean,std,min,max --overwrite --handle-dup-steps mean
  # --handle-dup-steps resolves the situation with two values for the same timestep
  # (occurs when continuing the training with current settings)
done