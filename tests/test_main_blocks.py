"""Run the self-tests embedded in ``if __name__ == "__main__"`` blocks.

The project keeps its unit/integration checks in the ``__main__`` section of
the modules they exercise.  Each entry below is executed as a separate Python
process so that one failing self-test cannot hide the results of another one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files whose ``__main__`` block contains a self-test (assertions and/or an
# explicit numerical comparison).  Long-running experiments such as
# ``sensitivity_analysis/dynamic_similarity/empirical.py`` are intentionally
# not included; they generate plots or sweep large parameter grids.
SELF_TEST_FILES = (
    "race_car/utils/track.py",
    "race_car/utils/config.py",
    "race_car/utils/scaling.py",
    "race_car/model.py",
    "race_car/acados_ocp.py",
    "race_car/controller.py",
    "race_car/env.py",
    "sensitivity_analysis/closed_loop_performance/sensitivity_utils.py",
)

MAX_OUTPUT_CHARS = 12_000
TIMEOUT_SECONDS = 900


def _tail(text: str) -> str:
    """Keep the end of long logs, where the actual error usually is."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_OUTPUT_CHARS
    return f"... ({omitted} characters omitted) ...\n{text[-MAX_OUTPUT_CHARS:]}"


def _format_output(stdout: str, stderr: str) -> str:
    return _tail(f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}")


@pytest.mark.parametrize(
    "relative_path",
    SELF_TEST_FILES,
    ids=[path.replace("/", "::") for path in SELF_TEST_FILES],
)
def test_main_block(relative_path: str) -> None:
    """Execute a module's ``__main__`` self-test and require a clean exit."""
    script = REPO_ROOT / relative_path
    assert script.is_file(), f"Self-test file is missing: {script}"

    env = os.environ.copy()
    # Keep matplotlib headless.  Some self-tests pause with input() to keep
    # their plots open; feed them newlines so CI never blocks.
    env.pop("DISPLAY", None)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            env=env,
            input="\n" * 20,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        pytest.fail(
            f"{relative_path} timed out after {TIMEOUT_SECONDS}s\n\n"
            f"{_format_output(stdout, stderr)}"
        )

    output = _format_output(result.stdout, result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"{relative_path} failed with exit code {result.returncode}\n\n{output}"
        )

    # race_car/model.py reports a mismatch via print() rather than assert.
    if "Results do not match" in result.stdout:
        pytest.fail(f"{relative_path} reported a numerical mismatch\n\n{output}")
