#!/bin/bash
# Convenience wrapper around the Optuna HPO driver.
#
# Examples:
#   race_car/scripts/run_hpo_sac_zop.sh --size default --dimensionless --n-trials 12
#   race_car/scripts/run_hpo_sac_zop.sh --size large --search-space full --n-jobs 4
#
# Any argument is forwarded to `python -m race_car.hpo.tune`.
# Run this script from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m race_car.hpo.tune "$@"
