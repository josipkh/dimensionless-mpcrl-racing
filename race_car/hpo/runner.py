"""Run one HPO trial as an isolated subprocess and stream validation scores.

A trial is a normal ``race_car/scripts/run_sac_zop.py`` invocation with a
per-trial output directory and a YAML file of config overrides.  Running each
configuration in its own process isolates the acados/PyTorch state between
trials, makes pruning as simple as terminating the process, and lets a single
crashed run be marked as failed without taking down the study.
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


class TrialExecutionError(RuntimeError):
    """The trial process exited unsuccessfully or produced no validation score."""


class TrialTimeoutError(TrialExecutionError):
    """The trial exceeded its wall-clock limit."""


class TrialPrunedError(RuntimeError):
    """The pruner asked to stop this trial early."""


def _read_new_validation_rows(
    path: Path, n_seen: int
) -> tuple[list[tuple[int, float]], int]:
    """Read newly appended rows from a leap-c ``val_log.csv``.

    The file is re-read on every poll because it only contains a handful of
    rows.  Incomplete trailing rows are ignored and picked up on the next poll.
    """
    if not path.is_file():
        return [], n_seen

    rows: list[tuple[int, float]] = []
    total_valid = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                step = int(float(row["timestamp"]))
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if total_valid >= n_seen:
                rows.append((step, score))
            total_valid += 1

    return rows, total_valid


def _signal_process(process: subprocess.Popen, sig: signal.Signals) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except ProcessLookupError:
            pass
    else:  # pragma: no cover - the project targets Linux, but be safe
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()


def _terminate_process(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    """Terminate a process and its children, escalating gracefully to SIGKILL.

    SIGINT is sent first so that Python can run ``finally``/``atexit`` handlers
    and remove acados' temporary code-generation directories.  SIGTERM and
    finally SIGKILL are fallbacks for a stuck child.
    """
    if process.poll() is not None:
        return

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        _signal_process(process, sig)
        try:
            process.wait(timeout=grace_s / 3)
            return
        except subprocess.TimeoutExpired:
            continue

    # SIGKILL was delivered, so the child is guaranteed to die; reap it to
    # avoid leaving a zombie behind.
    process.wait()


def choose_objective(
    rows: Sequence[tuple[int, float]], objective: str
) -> float:
    """Compute the trial objective from validation ``(step, score)`` rows."""
    if not rows:
        raise TrialExecutionError("Trial produced no validation scores.")

    if objective == "best":
        return max(score for _, score in rows)
    if objective == "final":
        return rows[-1][1]
    raise ValueError(f"Unknown objective '{objective}'. Use 'best' or 'final'.")


def run_trial(
    command: Sequence[str],
    trial_dir: Path,
    on_validation: Callable[[int, float], bool],
    *,
    poll_interval_s: float = 1.0,
    timeout_s: float | None = None,
    objective: str = "best",
    log_path: Path | None = None,
) -> tuple[float, list[tuple[int, float]]]:
    """Execute one trial and return its objective and validation history.

    Args:
        command: Command to run, usually ``python run_sac_zop.py ...``.
        trial_dir: Directory receiving ``val_log.csv`` and other run artifacts.
        on_validation: Called for every newly observed validation score.  Return
            ``True`` to prune the trial (the process is terminated).
        poll_interval_s: How often to poll the subprocess and validation log.
        timeout_s: Optional wall-clock limit for the whole trial.
        objective: ``"best"`` (max validation score) or ``"final"``.
        log_path: Where to redirect the subprocess output.  Defaults to
            ``trial_dir / "hpo_subprocess.log"``.

    Returns:
        ``(objective_value, validation_rows)``.

    Raises:
        TrialPrunedError: If ``on_validation`` requested pruning.
        TrialTimeoutError: If ``timeout_s`` was exceeded.
        TrialExecutionError: If the process failed or produced no scores.
    """
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path is not None else trial_dir / "hpo_subprocess.log"
    val_log = trial_dir / "val_log.csv"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    rows: list[tuple[int, float]] = []
    n_seen = 0
    start = time.monotonic()

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
                new_rows, n_seen = _read_new_validation_rows(val_log, n_seen)
                for step, score in new_rows:
                    rows.append((step, score))
                    if on_validation(step, score):
                        _terminate_process(process)
                        raise TrialPrunedError(
                            f"Trial pruned after validation at step {step}."
                        )

                return_code = process.poll()
                if return_code is not None:
                    break

                elapsed = time.monotonic() - start
                if timeout_s is not None and elapsed > timeout_s:
                    _terminate_process(process)
                    raise TrialTimeoutError(
                        f"Trial exceeded its time limit of {timeout_s:.0f} s."
                    )

                time.sleep(poll_interval_s)

        except BaseException:
            _terminate_process(process)
            raise

    # Pick up rows written between the last poll and process exit.
    new_rows, n_seen = _read_new_validation_rows(val_log, n_seen)
    rows.extend(new_rows)

    if process.returncode != 0:
        raise TrialExecutionError(
            f"Trial process exited with code {process.returncode}. "
            f"See {log_path}."
        )

    value = choose_objective(rows, objective)
    return value, rows


if __name__ == "__main__":
    # Smoke test: run a tiny command that writes a val_log.csv.
    import sys
    from tempfile import mkdtemp

    tmp = Path(mkdtemp())
    val_log = tmp / "val_log.csv"
    script = (
        "import csv, time;"
        f"f=open({str(val_log)!r}, 'w', newline='');"
        "w=csv.writer(f); w.writerow(['timestamp','score']); f.flush();"
        "[(w.writerow([s, -100+s]), f.flush(), time.sleep(0.05)) for s in (0, 5, 10)]"
    )
    observed: list[tuple[int, float]] = []
    value, rows = run_trial(
        [sys.executable, "-c", script],
        tmp,
        lambda step, score: observed.append((step, score)) or False,
        poll_interval_s=0.01,
    )
    assert value == -90.0, (value, rows)
    assert observed == [(0, -100.0), (5, -95.0), (10, -90.0)], observed

    # A pruning request must terminate the child and raise TrialPrunedError.
    prune_dir = tmp / "prune"
    prune_log = prune_dir / "val_log.csv"
    slow_script = (
        "import csv, time;"
        f"f=open({str(prune_log)!r}, 'w', newline='');"
        "w=csv.writer(f); w.writerow(['timestamp','score']);"
        "w.writerow([0, -100]); f.flush(); time.sleep(30)"
    )
    try:
        run_trial(
            [sys.executable, "-c", slow_script],
            prune_dir,
            lambda step, score: True,
            poll_interval_s=0.01,
        )
    except TrialPrunedError:
        pass
    else:
        raise AssertionError("Expected TrialPrunedError")

    print("runner self-test passed")
