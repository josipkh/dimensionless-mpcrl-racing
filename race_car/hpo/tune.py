"""Optuna HPO driver for the race car SAC-ZOP experiments.

Examples:
    # Small search space, 20 trials, four trials in parallel (default on 8 cores)
    python -m race_car.hpo.tune --size default --dimensionless --search-space small

    # Full search space, resume an existing study
    python -m race_car.hpo.tune --size default --dimensionless --search-space full \
        --study-name sac_zop_default_dimensionless_full --load-if-exists

    # Inspect a study and export the best configuration
    python -m race_car.hpo.report --study-dir output/hpo/<study_name>

The driver launches one ``run_sac_zop.py`` process per trial.  Validation
scores are streamed from the trial's ``val_log.csv`` into Optuna, which lets
multi-fidelity pruning terminate bad runs early.  Study metadata (database,
search space, settings, best config) is kept under ``output/hpo/<study>`` while
the potentially large per-trial artifacts go to ``/tmp/hpo/<study>`` by
default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import yaml

from race_car.hpo import runner
from race_car.hpo.config import dump_overrides, load_overrides, to_plain
from race_car.hpo.search_space import (
    ParamSpec,
    dump_space,
    expand_params,
    load_space,
    sample_params,
)
from race_car.hpo.summary import (
    print_best,
    state_counts,
    write_best_outputs,
    write_trials_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "race_car" / "scripts" / "run_sac_zop.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for SAC-ZOP on the race car environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # experiment setup (mirrors run_sac_zop.py)
    parser.add_argument("--size", choices=["default", "large"], default="default")
    parser.add_argument(
        "--dimensionless",
        action="store_true",
        help="Use the dimensionless formulation (same flag as run_sac_zop.py).",
    )
    parser.add_argument("--train-steps", type=int, default=50_000)

    # study / budget
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL or path to a .db file. Defaults to <study>/study.db.",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Resume an existing study with the same name instead of failing.",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="Number of trials to run in parallel.",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help="Wall-clock budget for launching/optimising trials.",
    )
    parser.add_argument(
        "--trial-timeout-min",
        type=float,
        default=None,
        help="Kill an individual trial after this many minutes.",
    )

    # search space and HPO method
    parser.add_argument(
        "--search-space",
        type=str,
        default="full",
        help="Preset name under race_car/hpo/search_spaces or a YAML path.",
    )
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument(
        "--pruner", choices=["hyperband", "median", "none"], default="hyperband"
    )
    parser.add_argument(
        "--min-resource",
        type=int,
        default=5_000,
        help="First rung (environment steps) for HyperbandPruner.",
    )
    parser.add_argument("--reduction-factor", type=int, default=3)

    # objective
    parser.add_argument(
        "--objective",
        choices=["best", "final"],
        default="best",
        help=(
            "Validation metric to maximise. 'best' is the best checkpoint; "
            "'final' is the last validation score."
        ),
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=2_500,
        help="Validation interval in environment steps inside each trial.",
    )
    parser.add_argument(
        "--val-num-rollouts",
        type=int,
        default=3,
        help="Validation episodes averaged at each checkpoint (reduces noise).",
    )
    parser.add_argument(
        "--tuning-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Training seeds cycled over trials. Use held-out seeds for test runs.",
    )

    # outputs and overrides
    parser.add_argument("--output-root", type=str, default="output/hpo")
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="/tmp/hpo",
        help="Root for large per-trial run artifacts.",
    )
    parser.add_argument(
        "--base-overrides",
        type=str,
        default=None,
        help="YAML file with fixed config overrides applied to every trial.",
    )
    parser.add_argument(
        "--keep-best-artifacts",
        action="store_true",
        help="Copy the best trial's small log files into <study>/best_trial.",
    )
    parser.add_argument(
        "--cleanup-artifacts",
        action="store_true",
        help="Delete non-best trial directories under --artifacts-root at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and print configurations without launching training runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep the trainer's verbose logging inside each trial.",
    )
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--progress", action="store_true")
    return parser


def _storage_url(storage: str | None, study_dir: Path) -> str:
    if storage is None:
        return f"sqlite:///{(study_dir / 'study.db').resolve()}"
    if "://" in storage:
        return storage
    path = Path(storage).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _make_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.sampler_seed)
    return optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        multivariate=True,
        group=True,
        n_startup_trials=5,
        n_ei_candidates=24,
    )


def _make_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    if args.min_resource >= args.train_steps:
        raise ValueError(
            f"--min-resource ({args.min_resource}) must be smaller than "
            f"--train-steps ({args.train_steps}) for HyperbandPruner."
        )
    return optuna.pruners.HyperbandPruner(
        min_resource=args.min_resource,
        max_resource=args.train_steps,
        reduction_factor=args.reduction_factor,
    )


def _study_name(args: argparse.Namespace) -> str:
    formulation = "dimensionless" if args.dimensionless else "dimensional"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sac_zop_{args.size}_{formulation}_{timestamp}"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            to_plain(data), file, sort_keys=False, default_flow_style=False
        )


def _check_or_write_space(specs: list[ParamSpec], study_dir: Path) -> None:
    path = study_dir / "search_space.yaml"
    if path.exists():
        existing, _ = load_space(path)
        existing_dict = {spec.name: spec.to_dict() for spec in existing}
        new_dict = {spec.name: spec.to_dict() for spec in specs}
        if existing_dict != new_dict:
            raise RuntimeError(
                f"The search space in '{path}' differs from the requested one. "
                "Start a new study (or use the same search-space file) to keep "
                "the study comparable."
            )
        return
    dump_space(specs, path)


def _append_invocation(study_dir: Path, record: dict[str, Any]) -> None:
    path = study_dir / "invocations.yaml"
    records: list[Any] = []
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and isinstance(data.get("invocations"), list):
            records = data["invocations"]
    records.append(record)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            to_plain(records), file, sort_keys=False, default_flow_style=False
        )


def _dry_run(args: argparse.Namespace, specs: list[ParamSpec]) -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=args.sampler_seed))
    print(f"Search space: {args.search_space} ({len(specs)} parameters)")
    for index in range(args.n_trials):
        trial = study.ask()
        params = sample_params(trial, specs)
        print(f"\n# trial {index}")
        print(yaml.safe_dump(params, sort_keys=False, default_flow_style=False))


def _trial_command(
    args: argparse.Namespace,
    seed: int,
    trial_dir: Path,
    overrides_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(RUN_SCRIPT),
        "--size",
        args.size,
        "--seed",
        str(seed),
        "--train_steps",
        str(args.train_steps),
        "--output_path",
        str(trial_dir),
        "--config_overrides",
        str(overrides_path),
    ]
    if args.dimensionless:
        command.append("--dimensionless")
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    specs, space_path = load_space(args.search_space)
    study_name = args.study_name or _study_name(args)
    study_dir = Path(args.output_root).expanduser() / study_name
    artifacts_dir = Path(args.artifacts_root).expanduser() / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _dry_run(args, specs)
        return 0

    _check_or_write_space(specs, study_dir)

    base_overrides = load_overrides(args.base_overrides) if args.base_overrides else {}
    fixed_overrides = {
        "val_report_score": args.objective,
        "val_freq": args.val_freq,
        "val_num_rollouts": args.val_num_rollouts,
        "val_num_render_rollouts": 0,
        "val_render_mode": None,
        "ckpt_modus": "none",
        "log.csv_logger": True,
        "log.tensorboard_logger": False,
        "log.verbose": bool(args.verbose),
        "log.interval": args.log_interval,
    }

    overlap = set(base_overrides) & set(fixed_overrides)
    if overlap:
        print(
            "Note: the following base overrides are controlled by the HPO driver "
            f"and will be replaced: {sorted(overlap)}"
        )
    sampled_targets = {target for spec in specs for target in spec.targets}
    overlap = set(base_overrides) & sampled_targets
    if overlap:
        print(
            "Note: the following base overrides coincide with sampled targets; "
            f"sampled values will win: {sorted(overlap)}"
        )

    storage = _storage_url(args.storage, study_dir)
    settings = {
        "study_name": study_name,
        "storage": storage,
        "size": args.size,
        "dimensionless": args.dimensionless,
        "train_steps": args.train_steps,
        "objective": args.objective,
        "val_freq": args.val_freq,
        "val_num_rollouts": args.val_num_rollouts,
        "tuning_seeds": list(args.tuning_seeds),
        "search_space": str(space_path),
        "base_overrides": base_overrides,
        "fixed_overrides": fixed_overrides,
        "sampler": args.sampler,
        "sampler_seed": args.sampler_seed,
        "pruner": args.pruner,
        "min_resource": args.min_resource,
        "reduction_factor": args.reduction_factor,
        "n_trials": args.n_trials,
        "n_jobs": args.n_jobs,
    }
    _write_yaml(settings, study_dir / "run_settings.yaml")

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=_make_sampler(args),
            pruner=_make_pruner(args),
            direction="maximize",
            load_if_exists=args.load_if_exists,
        )
    except optuna.exceptions.DuplicatedStudyError as exc:
        raise SystemExit(
            f"A study named '{study_name}' already exists in '{storage}'. "
            "Pass --load-if-exists to continue it."
        ) from exc

    if not (study_dir / "study_meta.yaml").exists():
        _write_yaml(
            {
                "study_name": study_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "storage": storage,
                "search_space_file": str(space_path),
                "n_parameters": len(specs),
            },
            study_dir / "study_meta.yaml",
        )

    _append_invocation(
        study_dir,
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "argv": [sys.executable, "-m", "race_car.hpo.tune", *sys.argv[1:]],
            "n_trials": args.n_trials,
            "n_jobs": args.n_jobs,
        },
    )

    timeout_s = args.timeout_hours * 3600 if args.timeout_hours else None
    trial_timeout_s = (
        args.trial_timeout_min * 60 if args.trial_timeout_min else None
    )

    print(f"Study directory : {study_dir}")
    print(f"Artifacts root  : {artifacts_dir}")
    print(f"Storage         : {storage}")
    print(f"Search space    : {space_path} ({len(specs)} parameters)")
    print(f"Objective       : maximise {args.objective} validation score")
    print(
        f"Budget          : {args.n_trials} trials, {args.n_jobs} in parallel, "
        f"{args.train_steps} steps per trial"
    )

    def objective(trial: optuna.Trial) -> float:
        started = time.monotonic()
        seed = int(args.tuning_seeds[trial.number % len(args.tuning_seeds)])

        sampled = sample_params(trial, specs)
        overrides = {**base_overrides, **sampled, **fixed_overrides}

        trial_dir = artifacts_dir / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            # A fresh process should not silently resume an old checkpoint.
            shutil.rmtree(trial_dir, ignore_errors=True)
        trial_dir.mkdir(parents=True, exist_ok=True)
        overrides_path = trial_dir / "config_overrides.yaml"
        dump_overrides(overrides, overrides_path)
        dump_overrides(sampled, trial_dir / "sampled_params.yaml")

        trial.set_user_attr("seed", seed)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("train_steps", args.train_steps)
        trial.set_user_attr("overrides", overrides)

        best_so_far = float("-inf")

        def on_validation(step: int, score: float) -> bool:
            nonlocal best_so_far
            best_so_far = max(best_so_far, score)
            value = best_so_far if args.objective == "best" else score
            trial.report(value, step=step)
            return trial.should_prune()

        try:
            value, rows = runner.run_trial(
                _trial_command(args, seed, trial_dir, overrides_path),
                trial_dir,
                on_validation,
                poll_interval_s=1.0,
                timeout_s=trial_timeout_s,
                objective=args.objective,
            )
        except runner.TrialPrunedError:
            raise optuna.TrialPruned()
        finally:
            trial.set_user_attr(
                "duration_s", round(time.monotonic() - started, 1)
            )

        trial.set_user_attr("n_validations", len(rows))
        return value

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=timeout_s,
        n_jobs=args.n_jobs,
        catch=(runner.TrialExecutionError,),
        show_progress_bar=args.progress,
    )

    print(f"\nTrials by state: {dict(state_counts(study))}")
    write_trials_csv(study, study_dir / "trials.csv")

    best = write_best_outputs(
        study,
        specs,
        study_dir,
        base_overrides=base_overrides,
        fixed_overrides=fixed_overrides,
        copy_artifacts=args.keep_best_artifacts,
    )
    if best is not None:
        print_best(best, specs)
        print(f"\nBest overrides written to: {study_dir / 'best_overrides.yaml'}")
        print("Re-run or evaluate on held-out seeds with:")
        flags = " --dimensionless" if args.dimensionless else ""
        print(
            f"  python race_car/scripts/run_sac_zop.py --size {args.size}{flags} "
            f"--seed 5 --config_overrides "
            f"{study_dir / 'best_overrides.yaml'}"
        )

    if args.cleanup_artifacts:
        best_dir = (
            Path(str(best.user_attrs.get("trial_dir", ""))) if best is not None else None
        )
        for trial in study.trials:
            directory = Path(str(trial.user_attrs.get("trial_dir", "")))
            if directory.is_dir() and directory != best_dir:
                shutil.rmtree(directory, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
