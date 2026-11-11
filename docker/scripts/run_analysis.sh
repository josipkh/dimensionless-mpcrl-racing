#!/usr/bin/env bash
# Run the test suite and the deterministic sensitivity analyses in isolated
# working directories.  Results are written to /workspace/results by default.
set -Eeuo pipefail

ROOT="${REPO_ROOT:-/workspace}"
cd "$ROOT"

if [[ -n "${REPRO_RESULTS:-}" ]]; then
    RESULTS_ROOT="$REPRO_RESULTS"
    mkdir -p "$RESULTS_ROOT"
else
    STAMP="${REPRO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    RESULTS_ROOT="$ROOT/results/$STAMP"
    mkdir -p "$RESULTS_ROOT"
    ln -sfn "$STAMP" "$ROOT/results/latest"
fi

RESULTS_ROOT="$(cd "$RESULTS_ROOT" && pwd)"
export RESULTS_ROOT

{
    echo "run_id: $(basename "$RESULTS_ROOT")"
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit: ${GIT_COMMIT:-unknown}"
    echo "git_dirty: ${GIT_DIRTY:-unknown}"
    echo "acados_source_dir: ${ACADOS_SOURCE_DIR:-unknown}"
    echo "python: $(python --version 2>&1)"
    echo "platform: $(uname -a)"
    echo
    echo "--- Python packages ---"
    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("numpy", "scipy", "casadi", "torch", "leap-c", "acados_template"):
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}: not installed")
PY
    echo
    echo "--- Environment ---"
    env | sort | grep -E '^(ACADOS|LD_LIBRARY_PATH|MPLBACKEND|OMP_|OPENBLAS_|MKL_|PYTHONPATH|GIT_)' || true
} > "$RESULTS_ROOT/provenance.txt"

echo "Results directory: $RESULTS_ROOT"
echo

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
echo "==> pytest"
(cd "$ROOT" && pytest -v) 2>&1 | tee "$RESULTS_ROOT/tests.log"
echo

# ---------------------------------------------------------------------------
# Sensitivity analyses
# ---------------------------------------------------------------------------
# Run only a subset with, e.g.:
#   ANALYSIS_STEPS=dynamic-analytical,compare-gradients
ANALYSIS_STEPS="${ANALYSIS_STEPS:-dynamic-analytical,dynamic-empirical,closed-loop-analytical,closed-loop-empirical,compare-gradients}"

declare -A SCRIPTS=(
    [dynamic-analytical]="sensitivity_analysis/dynamic_similarity/analytical.py"
    [dynamic-empirical]="sensitivity_analysis/dynamic_similarity/empirical.py"
    [closed-loop-analytical]="sensitivity_analysis/closed_loop_performance/analytical.py"
    [closed-loop-empirical]="sensitivity_analysis/closed_loop_performance/empirical.py"
    [compare-gradients]="sensitivity_analysis/closed_loop_performance/compare_gradients.py"
)

run_step() {
    local name="$1"
    local script="${SCRIPTS[$name]:-}"

    if [[ -z "$script" ]]; then
        echo "Unknown analysis step: $name" >&2
        exit 2
    fi
    if [[ ",$ANALYSIS_STEPS," != *",$name,"* ]]; then
        echo "==> $name (skipped)"
        return 0
    fi

    local step_dir="$RESULTS_ROOT/$name"
    mkdir -p "$step_dir/output"

    echo "==> $name"
    echo "    script: $script"
    echo "    output: $step_dir"

    # Run from the step directory so that the scripts' relative `output/`
    # paths are isolated and cannot reuse stale cached .npy/.npz files.
    (
        cd "$step_dir"
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        MPLBACKEND=Agg \
            python "$ROOT/$script"
    ) 2>&1 | tee "$step_dir/run.log"

    echo
}

for step in dynamic-analytical dynamic-empirical closed-loop-analytical closed-loop-empirical compare-gradients; do
    run_step "$step"
done

echo "Analysis complete."
echo "Results: $RESULTS_ROOT"
