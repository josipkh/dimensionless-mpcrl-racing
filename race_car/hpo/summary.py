"""Helpers for summarising an Optuna study and exporting its best configuration."""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from race_car.hpo.config import dump_overrides, to_plain
from race_car.hpo.search_space import ParamSpec, expand_params

BEST_ARTIFACTS = (
    "val_log.csv",
    "config.yaml",
    "config_overrides.yaml",
    "output.log",
    "hpo_subprocess.log",
    "git.txt",
    "my_git_diff.txt",
)


def state_counts(study: Any) -> Counter[str]:
    """Count trials by Optuna state."""
    return Counter(str(trial.state.name) for trial in study.trials)


def write_best_outputs(
    study: Any,
    specs: Iterable[ParamSpec],
    study_dir: Path,
    *,
    base_overrides: Mapping[str, Any] | None = None,
    fixed_overrides: Mapping[str, Any] | None = None,
    copy_artifacts: bool = False,
) -> Any | None:
    """Write ``best_overrides.yaml`` and ``best_trial.yaml`` for a study.

    Returns the best trial, or ``None`` when the study has no complete trials.
    """
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    try:
        best = study.best_trial
    except ValueError:
        print("No complete trials yet; skipping best-configuration export.")
        return None

    sampled = expand_params(specs, dict(best.params))
    overrides = {
        **(base_overrides or {}),
        **sampled,
        **(fixed_overrides or {}),
    }

    best_overrides_path = study_dir / "best_overrides.yaml"
    dump_overrides(overrides, best_overrides_path)

    metadata = {
        "study_name": study.study_name,
        "trial_number": best.number,
        "objective_value": best.value,
        "state": best.state.name,
        "params_sampled": dict(best.params),
        "params_config": sampled,
        "user_attrs": dict(best.user_attrs),
        "n_trials": len(study.trials),
        "trial_states": dict(state_counts(study)),
    }
    with (study_dir / "best_trial.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            to_plain(metadata), file, sort_keys=False, default_flow_style=False
        )

    if copy_artifacts:
        source = Path(str(best.user_attrs.get("trial_dir", "")))
        if source.is_dir():
            destination = study_dir / "best_trial"
            destination.mkdir(parents=True, exist_ok=True)
            for name in BEST_ARTIFACTS:
                src = source / name
                if src.is_file():
                    shutil.copy2(src, destination / name)

    return best


def write_trials_csv(study: Any, path: str | Path) -> None:
    """Write one row per trial to CSV without requiring pandas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    param_names: list[str] = []
    for trial in study.trials:
        for name in trial.params:
            if name not in param_names:
                param_names.append(name)

    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_s",
        "seed",
        "train_steps",
        "n_validations",
        "trial_dir",
        *[f"param.{name}" for name in param_names],
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row: dict[str, Any] = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
                "duration_s": trial.user_attrs.get("duration_s"),
                "seed": trial.user_attrs.get("seed"),
                "train_steps": trial.user_attrs.get("train_steps"),
                "n_validations": trial.user_attrs.get("n_validations"),
                "trial_dir": trial.user_attrs.get("trial_dir"),
            }
            for name in param_names:
                row[f"param.{name}"] = trial.params.get(name)
            writer.writerow(row)


def print_best(best: Any, specs: Iterable[ParamSpec]) -> None:
    """Print a concise summary of the best trial."""
    sampled = expand_params(specs, dict(best.params))
    print("\n" + "=" * 70)
    print(f"Best trial: #{best.number}  value={best.value:.4f}")
    print("-" * 70)
    for key, value in sampled.items():
        print(f"  {key}: {value}")
    duration = best.user_attrs.get("duration_s")
    if duration is not None:
        print(f"  duration_s: {duration:.1f}")
    seed = best.user_attrs.get("seed")
    if seed is not None:
        print(f"  tuning_seed: {seed}")
    print("=" * 70)
