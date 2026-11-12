"""Summarise an Optuna study created by :mod:`race_car.hpo.tune`.

This loads the SQLite study, prints the best configuration, writes a plain-text
report and interactive Plotly HTML figures, and refreshes
``best_overrides.yaml`` so the tuned configuration can be re-run directly.

Example:
    python -m race_car.hpo.report --study-dir output/hpo/<study_name>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import optuna
import yaml

from race_car.hpo.search_space import expand_params, load_space
from race_car.hpo.summary import (
    print_best,
    state_counts,
    write_best_outputs,
    write_trials_csv,
)

DEFAULT_ROOT = Path("output/hpo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report and plot an Optuna HPO study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--study-dir",
        type=str,
        default=None,
        help="Study directory. Defaults to the most recently modified study.",
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip Plotly HTML figures."
    )
    return parser


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def _latest_study_dir(root: Path) -> Path:
    candidates = [path.parent for path in root.glob("*/study.db")]
    if not candidates:
        raise SystemExit(f"No studies with a study.db found under '{root}'.")
    return max(candidates, key=lambda path: (path / "study.db").stat().st_mtime)


def _load_study(args: argparse.Namespace, study_dir: Path) -> optuna.Study:
    meta = _read_yaml(study_dir / "study_meta.yaml")
    settings = _read_yaml(study_dir / "run_settings.yaml")

    local_storage = f"sqlite:///{(study_dir / 'study.db').resolve()}"
    storage = args.storage or settings.get("storage") or local_storage

    # A study directory copied to another machine keeps an absolute storage
    # path in run_settings.yaml.  Fall back to the local database when that
    # path is gone but the study directory still contains study.db.
    if args.storage is None and storage.startswith("sqlite:///"):
        recorded = Path(storage.removeprefix("sqlite:///"))
        if not recorded.is_file() and (study_dir / "study.db").is_file():
            storage = local_storage

    study_name = args.study_name or meta.get("study_name")
    return optuna.load_study(study_name=study_name, storage=storage)


def _format_report(study: optuna.Study, specs: Any, top_k: int) -> str:
    counts = Counter(str(trial.state.name) for trial in study.trials)
    lines = [
        f"Study: {study.study_name}",
        f"Direction: {study.direction.name}",
        f"Trials: {len(study.trials)}",
        f"States: {dict(counts)}",
    ]

    complete = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        lines.append("No complete trials yet.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Top {min(top_k, len(complete))} complete trials:")
    header = f"{'trial':>6} {'value':>12} {'seed':>6} {'duration_s':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for trial in sorted(complete, key=lambda item: item.value or float("-inf"), reverse=True)[
        :top_k
    ]:
        lines.append(
            f"{trial.number:>6} {trial.value:>12.4f} "
            f"{str(trial.user_attrs.get('seed', '-')):>6} "
            f"{str(trial.user_attrs.get('duration_s', '-')):>12}"
        )

    try:
        best = study.best_trial
        expanded = expand_params(specs, dict(best.params))
        lines.append("")
        lines.append(f"Best trial: #{best.number} (value={best.value:.4f})")
        for key, value in expanded.items():
            lines.append(f"  {key}: {value}")
    except ValueError:
        pass

    try:
        importances = optuna.importance.get_param_importances(study)
        lines.append("")
        lines.append("fANOVA parameter importance (complete trials):")
        for name, importance in sorted(
            importances.items(), key=lambda item: item[1], reverse=True
        ):
            lines.append(f"  {name}: {importance:.4f}")
    except Exception as exc:  # too few trials, constant params, etc.
        lines.append("")
        lines.append(f"Parameter importance unavailable: {exc}")

    return "\n".join(lines)


def _write_plots(study: optuna.Study, study_dir: Path) -> None:
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )
    except ImportError as exc:  # pragma: no cover
        print(f"Plotting unavailable: {exc}")
        return

    plots = {
        "optimization_history.html": plot_optimization_history,
        "param_importances.html": plot_param_importances,
        "parallel_coordinate.html": plot_parallel_coordinate,
    }
    for filename, plot_fn in plots.items():
        try:
            figure = plot_fn(study)
            figure.write_html(study_dir / filename, include_plotlyjs="cdn")
        except Exception as exc:  # pragma: no cover - plotting is best effort
            print(f"Could not create {filename}: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser()
    study_dir = (
        Path(args.study_dir).expanduser() if args.study_dir else _latest_study_dir(root)
    )
    study_dir = study_dir.resolve()

    study = _load_study(args, study_dir)
    space_file = _read_yaml(study_dir / "run_settings.yaml").get("search_space")
    if not space_file:
        space_file = study_dir / "search_space.yaml"
    specs, _ = load_space(space_file)

    report = _format_report(study, specs, args.top_k)
    print(report)
    (study_dir / "report.txt").write_text(report + "\n", encoding="utf-8")

    write_trials_csv(study, study_dir / "trials.csv")

    settings = _read_yaml(study_dir / "run_settings.yaml")
    best = write_best_outputs(
        study,
        specs,
        study_dir,
        base_overrides=settings.get("base_overrides", {}),
        fixed_overrides=settings.get("fixed_overrides", {}),
        copy_artifacts=False,
    )
    if best is not None:
        print_best(best, specs)

    if not args.no_plots:
        _write_plots(study, study_dir)
        print(f"\nPlots written to {study_dir}")

    print(f"Report written to {study_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
