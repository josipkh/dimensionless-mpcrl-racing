"""Optuna HPO for SAC-ZOP with a fixed MPC/policy architecture.

Only five trainer hyperparameters are searched::

    tau, lr_q, lr_pi, lr_alpha, init_alpha

Everything else (MPC parameters, direct parameter learning without hidden
layers, critic size, batch size, gamma, ...) is fixed by ``create_cfg`` in
``race_car/scripts/run_sac_zop.py``.

Each trial runs in its own subprocess.  Validation scores come from
``val_log.csv`` and the MPC solve success rate from ``val_policy_log.csv``.  A
trial is discarded as soon as a validation checkpoint reports a solve rate
below ``--min-solve-rate`` (default 85%), so the study selects configurations
that are both fast and reliable.  Median pruning stops clearly bad trials early.

Examples:
    python -m race_car.hpo.tune --size default --dimensionless --n-trials 20 --n-jobs 4
    python -m race_car.hpo.tune --study-name my_study --load-if-exists
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import optuna
import yaml

from race_car.hpo.overrides import dump_overrides

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "race_car" / "scripts" / "run_sac_zop.py"

# Search ranges are centred on the June 2026 baseline (tau=0.005,
# lr_q=lr_pi=lr_alpha=3e-4, init_alpha=0.005) and span roughly one decade in
# each direction.  Edit these tuples to widen or narrow the search.
SEARCH_RANGES: dict[str, tuple[float, float]] = {
    "tau": (1.0e-4, 1.0e-1),
    "lr_q": (1.0e-5, 1.0e-2),
    "lr_pi": (1.0e-5, 1.0e-2),
    "lr_alpha": (1.0e-5, 1.0e-2),
    "init_alpha": (1.0e-4, 1.0e-1),
}

# Fixed HPO-only settings: cheap logging, no rendering/checkpoints.  The trainer
# and controller configuration itself is left untouched.
FIXED_OVERRIDES: dict[str, object] = {
    "val_report_score": "best",
    "val_num_render_rollouts": 0,
    "val_render_mode": None,
    "ckpt_modus": "none",
    "log.csv_logger": True,
    "log.tensorboard_logger": False,
    "log.verbose": False,
}

BEST_ARTIFACTS = ("val_log.csv", "val_policy_log.csv", "config.yaml", "output.log")


class TrialExecutionError(RuntimeError):
    """The trial process failed or produced no usable validation data."""


class TrialTimeoutError(TrialExecutionError):
    """The trial exceeded its wall-clock limit."""


class DiscardedTrial(RuntimeError):
    """MPC solve rate fell below the reliability threshold."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for SAC-ZOP (five hyperparameters, MPC reliability filter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", choices=["default", "large"], default="default")
    parser.add_argument("--dimensionless", action="store_true")
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--val-freq", type=int, default=2_500)
    parser.add_argument("--val-num-rollouts", type=int, default=3)
    parser.add_argument(
        "--min-solve-rate",
        type=float,
        default=0.85,
        help="Discard a trial whose MPC solve rate falls below this value at any validation.",
    )

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="Number of trials trained in parallel.",
    )
    parser.add_argument(
        "--tuning-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Training seeds cycled over trials. Evaluate the winner on held-out seeds.",
    )
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--pruner", choices=["median", "none"], default="median")
    parser.add_argument("--trial-timeout-min", type=float, default=None)

    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="output/hpo")
    parser.add_argument("--artifacts-root", type=str, default="/tmp/hpo")
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Copy the best trial's small log files into <study>/best_trial.",
    )
    parser.add_argument(
        "--cleanup-artifacts",
        action="store_true",
        help="Delete non-best per-trial directories after the study finishes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def sample_params(trial: optuna.Trial) -> dict[str, float]:
    """Sample the five tuned hyperparameters."""
    return {
        name: trial.suggest_float(name, low, high, log=True)
        for name, (low, high) in SEARCH_RANGES.items()
    }


def _read_new_rows(path: Path, n_seen: int) -> tuple[list[dict[str, str]], int]:
    """Read new rows from a CSV that is being appended to by another process.

    An incomplete trailing line is ignored until it is terminated by a newline,
    which avoids counting a half-written validation row.
    """
    if not path.is_file():
        return [], n_seen
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    rows = list(csv.DictReader(text.splitlines()))
    return rows[n_seen:], len(rows)


def _kill(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    """Terminate the trial process group, escalating SIGINT -> SIGTERM -> SIGKILL."""
    if process.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_s / 3)
            return
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def run_trial(
    command: Sequence[str],
    trial_dir: Path,
    on_validation: Callable[[int, float], None],
    *,
    min_solve_rate: float,
    timeout_s: float | None = None,
    poll_interval_s: float = 1.0,
) -> tuple[list[tuple[int, float]], dict[int, float]]:
    """Run one training subprocess and stream its validation metrics.

    Args:
        command: Training command, normally a ``run_sac_zop.py`` invocation.
        trial_dir: Directory receiving the trial's logs.
        on_validation: Called with ``(step, score)`` for every validation whose
            MPC statistics are available.  It may raise ``optuna.TrialPruned``.
        min_solve_rate: Discard the trial if any checkpoint has a lower
            ``success_rate`` in ``val_policy_log.csv``.
        timeout_s: Optional wall-clock limit.
        poll_interval_s: How often the logs and process are polled.

    Returns:
        The validation ``(step, score)`` history and ``{step: success_rate}``.

    Raises:
        DiscardedTrial: MPC solve rate fell below ``min_solve_rate``.
        TrialTimeoutError: The wall-clock limit was exceeded.
        TrialExecutionError: The process failed or produced no validations.
    """
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    score_log = trial_dir / "val_log.csv"
    policy_log = trial_dir / "val_policy_log.csv"
    log_path = trial_dir / "hpo_subprocess.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    scores: list[tuple[int, float]] = []
    pending: list[tuple[int, float]] = []
    rates: dict[int, float] = {}
    n_scores = n_rates = 0
    start = time.monotonic()

    def drain() -> None:
        nonlocal n_scores, n_rates
        new_scores, n_scores = _read_new_rows(score_log, n_scores)
        new_rates, n_rates = _read_new_rows(policy_log, n_rates)
        for row in new_scores:
            try:
                pending.append((int(float(row["timestamp"])), float(row["score"])))
            except (KeyError, TypeError, ValueError):
                continue
        for row in new_rates:
            try:
                rates[int(float(row["timestamp"]))] = float(row["success_rate"])
            except (KeyError, TypeError, ValueError):
                continue
        while pending and pending[0][0] in rates:
            step, score = pending.pop(0)
            rate = rates[step]
            if rate < min_solve_rate:
                raise DiscardedTrial(
                    f"MPC solve rate {rate:.1%} is below {min_solve_rate:.0%} "
                    f"at step {step}."
                )
            scores.append((step, score))
            on_validation(step, score)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            while True:
                drain()
                if process.poll() is not None:
                    break
                if timeout_s is not None and time.monotonic() - start > timeout_s:
                    raise TrialTimeoutError(
                        f"Trial exceeded its {timeout_s / 60:.0f} min limit."
                    )
                time.sleep(poll_interval_s)
        except BaseException:
            _kill(process)
            raise

    # Pick up rows written between the last poll and process exit.
    drain()
    if process.returncode != 0:
        raise TrialExecutionError(
            f"Trial process exited with code {process.returncode}. See {log_path}."
        )
    if not scores:
        raise TrialExecutionError(f"Trial produced no validation scores. See {log_path}.")
    return scores, rates


def _write_yaml(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _write_trials_csv(study: optuna.Study, path: Path) -> None:
    param_names = list(SEARCH_RANGES)
    fields = [
        "number",
        "state",
        "value",
        "duration_s",
        "seed",
        "n_validations",
        "min_solve_rate",
        "discarded_reason",
        *param_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            row: dict[str, object] = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_s": trial.user_attrs.get("duration_s"),
                "seed": trial.user_attrs.get("seed"),
                "n_validations": trial.user_attrs.get("n_validations"),
                "min_solve_rate": trial.user_attrs.get("min_solve_rate"),
                "discarded_reason": trial.user_attrs.get("discarded_reason"),
            }
            row.update({name: trial.params.get(name) for name in param_names})
            writer.writerow(row)


def _print_summary(study: optuna.Study) -> None:
    print(f"\nTrials by state: {dict(Counter(t.state.name for t in study.trials))}")
    try:
        best = study.best_trial
    except ValueError:
        print("No completed trials yet.")
        return

    print(f"\nBest trial #{best.number}: value={best.value:.4f}")
    for name, value in best.params.items():
        print(f"  {name}: {value}")
    if best.user_attrs.get("min_solve_rate") is not None:
        print(f"  min_solve_rate: {best.user_attrs['min_solve_rate']:.3f}")

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:  # too few/constant trials; importance is optional
        return
    print("\nParameter importance:")
    for name, importance in sorted(importances.items(), key=lambda item: -item[1]):
        print(f"  {name}: {importance:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    formulation = "dimensionless" if args.dimensionless else "dimensional"
    study_name = args.study_name or (
        f"hpo_{args.size}_{formulation}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    study_dir = Path(args.output_root).expanduser() / study_name
    artifacts_dir = Path(args.artifacts_root).expanduser() / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(args.sampler_seed))
        for _ in range(args.n_trials):
            print(yaml.safe_dump(sample_params(study.ask()), sort_keys=False))
        return 0

    settings = {
        "arguments": vars(args),
        "search_ranges": SEARCH_RANGES,
        "fixed_overrides": FIXED_OVERRIDES,
        "objective": "best validation score, discarding trials below --min-solve-rate",
    }
    _write_yaml(settings, study_dir / "run_settings.yaml")

    storage = f"sqlite:///{(study_dir / 'study.db').resolve()}"
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(
                seed=args.sampler_seed, n_startup_trials=5
            ),
            pruner=(
                optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
                if args.pruner == "median"
                else optuna.pruners.NopPruner()
            ),
            direction="maximize",
            load_if_exists=args.load_if_exists,
        )
    except optuna.exceptions.DuplicatedStudyError as exc:
        raise SystemExit(
            f"Study '{study_name}' already exists in '{storage}'. "
            "Pass --load-if-exists to continue it."
        ) from exc

    print(f"Study     : {study_dir}")
    print(f"Artifacts : {artifacts_dir}")
    print(f"Budget    : {args.n_trials} trials, {args.n_jobs} in parallel, "
          f"{args.train_steps} steps each")
    print(f"Reliability: discard if MPC solve rate < {args.min_solve_rate:.0%}")

    trial_timeout_s = args.trial_timeout_min * 60 if args.trial_timeout_min else None

    def objective(trial: optuna.Trial) -> float:
        started = time.monotonic()
        seed = int(args.tuning_seeds[trial.number % len(args.tuning_seeds)])
        sampled = sample_params(trial)

        trial_dir = artifacts_dir / f"trial_{trial.number:04d}"
        shutil.rmtree(trial_dir, ignore_errors=True)
        trial_dir.mkdir(parents=True, exist_ok=True)
        overrides_path = trial_dir / "config_overrides.yaml"
        dump_overrides(
            {
                **sampled,
                "val_freq": args.val_freq,
                "val_num_rollouts": args.val_num_rollouts,
                **FIXED_OVERRIDES,
            },
            overrides_path,
        )

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

        def on_validation(step: int, score: float) -> None:
            trial.report(score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at step {step}.")

        trial.set_user_attr("seed", seed)
        trial.set_user_attr("trial_dir", str(trial_dir))
        try:
            scores, rates = run_trial(
                command,
                trial_dir,
                on_validation,
                min_solve_rate=args.min_solve_rate,
                timeout_s=trial_timeout_s,
            )
        except DiscardedTrial as exc:
            trial.set_user_attr("discarded_reason", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        finally:
            trial.set_user_attr("duration_s", round(time.monotonic() - started, 1))

        trial.set_user_attr("n_validations", len(scores))
        trial.set_user_attr("min_solve_rate", min(rates.values()) if rates else None)
        return max(score for _, score in scores)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        catch=(TrialExecutionError,),
    )

    _write_trials_csv(study, study_dir / "trials.csv")
    _print_summary(study)

    best: optuna.trial.FrozenTrial | None
    try:
        best = study.best_trial
    except ValueError:
        best = None

    if best is not None:
        dump_overrides(best.params, study_dir / "best_overrides.yaml")
        print(f"\nBest overrides written to {study_dir / 'best_overrides.yaml'}")
        flags = " --dimensionless" if args.dimensionless else ""
        print("Re-run on held-out seeds with:")
        print(
            f"  python race_car/scripts/run_sac_zop.py --size {args.size}{flags} "
            f"--seed 5 --config_overrides {study_dir / 'best_overrides.yaml'}"
        )

        if args.keep_best:
            source = Path(str(best.user_attrs.get("trial_dir", "")))
            destination = study_dir / "best_trial"
            destination.mkdir(parents=True, exist_ok=True)
            for name in BEST_ARTIFACTS:
                if (source / name).is_file():
                    shutil.copy2(source / name, destination / name)

        if args.cleanup_artifacts:
            for trial in study.trials:
                directory = Path(str(trial.user_attrs.get("trial_dir", "")))
                if directory.is_dir() and directory != Path(
                    str(best.user_attrs.get("trial_dir", ""))
                ):
                    shutil.rmtree(directory, ignore_errors=True)

    return 0


def _self_test() -> None:
    """Exercise sampling, log streaming and the reliability filter."""
    from tempfile import mkdtemp

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(0))
    params = sample_params(study.ask())
    assert set(params) == set(SEARCH_RANGES)
    assert all(low <= value <= high for name, (low, high) in SEARCH_RANGES.items()
               for value in [params[name]])

    tmp = Path(mkdtemp())
    ok_dir = tmp / "ok"
    ok_dir.mkdir()
    ok_script = (
        "import csv;"
        f"sf=open({str(ok_dir / 'val_log.csv')!r},'w',newline='');"
        "sw=csv.writer(sf);sw.writerow(['timestamp','score']);"
        f"pf=open({str(ok_dir / 'val_policy_log.csv')!r},'w',newline='');"
        "pw=csv.writer(pf);pw.writerow(['timestamp','success_rate']);"
        "[(sw.writerow([s,sc]),pw.writerow([s,r]),sf.flush(),pf.flush())"
        " for s,sc,r in [(0,-100,1.0),(5,-95,0.9),(10,-90,0.95)]]"
    )
    observed: list[tuple[int, float]] = []
    scores, rates = run_trial(
        [sys.executable, "-c", ok_script],
        ok_dir,
        lambda step, score: observed.append((step, score)),
        min_solve_rate=0.85,
        poll_interval_s=0.01,
    )
    assert observed == [(0, -100.0), (5, -95.0), (10, -90.0)]
    assert scores == observed
    assert rates == {0: 1.0, 5: 0.9, 10: 0.95}

    bad_dir = tmp / "bad"
    bad_dir.mkdir()
    bad_script = (
        "import csv;"
        f"sf=open({str(bad_dir / 'val_log.csv')!r},'w',newline='');"
        "sw=csv.writer(sf);sw.writerow(['timestamp','score']);"
        f"pf=open({str(bad_dir / 'val_policy_log.csv')!r},'w',newline='');"
        "pw=csv.writer(pf);pw.writerow(['timestamp','success_rate']);"
        "[(sw.writerow([s,sc]),pw.writerow([s,r]),sf.flush(),pf.flush())"
        " for s,sc,r in [(0,-100,1.0),(5,-95,0.5)]]"
    )
    try:
        run_trial(
            [sys.executable, "-c", bad_script],
            bad_dir,
            lambda step, score: None,
            min_solve_rate=0.85,
            poll_interval_s=0.01,
        )
    except DiscardedTrial:
        pass
    else:
        raise AssertionError("Expected DiscardedTrial for a low MPC solve rate")

    print("tune self-test passed")


if __name__ == "__main__":
    # ``python -m race_car.hpo.tune`` is the CLI; executing the file directly
    # (as tests/test_main_blocks.py does) runs the self-test.
    if __package__:
        raise SystemExit(main())
    _self_test()
