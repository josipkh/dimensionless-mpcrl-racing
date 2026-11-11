#!/usr/bin/env bash
# Entry point for the reproducibility container.
set -Eeuo pipefail

ROOT="${REPO_ROOT:-/workspace}"
MODE="${1:-analysis}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$MODE" in
    test|tests)
        exec pytest -v "$@"
        ;;
    analysis|analyse|analyze|sensitivity)
        exec "$ROOT/scripts/run_analysis.sh" "$@"
        ;;
    rl|train|experiments)
        exec "$ROOT/scripts/run_rl.sh" "$@"
        ;;
    full|all)
        "$ROOT/scripts/run_analysis.sh"
        exec "$ROOT/scripts/run_rl.sh" all "$@"
        ;;
    shell|bash)
        exec bash "$@"
        ;;
    help|-h|--help)
        cat <<'EOF'
Usage: docker compose run --rm reproduce [MODE] [ARGS...]

Modes:
  analysis   Run the test suite and all sensitivity analyses (default).
  test       Run only pytest.
  rl [SUITE] Run RL experiments: main (default), transfer, baseline, or all.
  full       Run analysis, then every RL experiment.
  shell      Open an interactive shell in the container.

Examples:
  make reproduce
  make rl
  docker compose run --rm reproduce test
  docker compose run --rm -e RL_SEEDS="0 1" -e RL_JOBS=2 reproduce rl main
EOF
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Run 'docker compose run --rm reproduce help' for available modes." >&2
        exit 2
        ;;
esac
