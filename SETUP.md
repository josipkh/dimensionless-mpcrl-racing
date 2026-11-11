# Local development setup

This note records how the project was set up and tested on a machine with
**Ubuntu 26.04**, **no `sudo`**, and **only Python 3.14** installed. The setup
matches the CI workflow in [`.github/workflows/tests.yml`](.github/workflows/tests.yml),
which uses Python 3.12.

## Why Python 3.12?

The pinned dependencies in [`requirements.txt`](requirements.txt) (for example
`scipy==1.15.2`) do not provide wheels for Python 3.14. We therefore use a
`uv`-managed Python 3.12 interpreter.

Versions used for verification:

| Tool | Version |
| --- | --- |
| Python | 3.12.16 |
| uv | 0.13.7 |
| CMake | 4.4.4 |
| Ninja | 1.13.2 |
| acados | v0.5.5 (`59d93e17`) |
| leap-c | `c8a7e3e` |

## 1. Prerequisites

If you have `sudo`, the usual system packages are enough:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake ninja-build wget
```

On this machine there was no `sudo`, so the already-installed `gcc`, `g++`,
`make`, and `wget` were used, and CMake/Ninja were installed as user-level
`uv` tools (see step 3).

## 2. Clone and initialize submodules

A fresh clone does **not** include the contents of `external/leap-c`, so
initialize the submodules recursively:

```bash
git clone <repository-url>
cd dimensionless-mpcrl-racing
git submodule update --init --recursive
```

Verify the pinned revisions:

```bash
git -C external/leap-c rev-parse HEAD
# c8a7e3edadbad98a3fd9455aefdd64e19468876c
git -C external/leap-c/external/acados rev-parse HEAD
# 59d93e17d2985fdd73fc58b8a83ed8f83a024171
```

## 3. Install uv, Python 3.12, CMake, and Ninja

Install `uv` into your user account (no root required):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Add that `PATH` line to `~/.bashrc` so new shells can find `uv`, `cmake`, etc.

Install Python 3.12 and the build tools:

```bash
uv python install 3.12
uv tool install cmake
uv tool install ninja
```

Check:

```bash
uv --version
cmake --version
ninja --version
```

## 4. Create the project virtual environment

From the repository root:

```bash
uv venv --python 3.12 .venv
```

Install the pinned Python packages:

```bash
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements.txt
```

`--index-strategy unsafe-best-match` is needed because
`requirements.txt` adds the PyTorch CPU index via `--extra-index-url`. That
index also contains some generic packages (for example `packaging`), and `uv`
otherwise refuses to consider the PyPI versions of those pins. Plain `pip`
does not have this issue.

## 5. Build acados and install Tera

```bash
export ACADOS_SOURCE_DIR="$PWD/external/leap-c/external/acados"
export ACADOS_INSTALL_DIR="$ACADOS_SOURCE_DIR"

cd "$ACADOS_SOURCE_DIR"
cmake -S . -B build \
  -DACADOS_WITH_OPENMP=ON \
  -DACADOS_NUM_THREADS=1
cmake --build build --target install -- -j"$(nproc)"

# Renders the generated acados code; installs bin/t_renderer.
bash .github/linux/install_tera.sh
cd -
```

Verify the build:

```bash
ls "$ACADOS_SOURCE_DIR/lib/libacados.so"
"$ACADOS_SOURCE_DIR/bin/t_renderer" --version
```

`Ninja` is optional here; adding `-G Ninja` to the `cmake` configure command
makes rebuilds faster.

## 6. Install the local Python interfaces

The acados Python interface and leap-c are installed from the pinned
submodules, not from PyPI. `--no-deps` keeps the versions in
`requirements.txt` authoritative.

```bash
uv pip install --python .venv/bin/python --no-deps \
  -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"

uv pip install --python .venv/bin/python --no-deps \
  -e external/leap-c

uv pip check --python .venv/bin/python
```

Expected result: `All installed packages are compatible`.

## 7. Shell environment

The tests and debugger need acados and the repository on their paths. Create
a `.envrc` in the repository root (it is already git-ignored):

```bash
# .envrc
export ACADOS_SOURCE_DIR="$PWD/external/leap-c/external/acados"
export ACADOS_INSTALL_DIR="$ACADOS_SOURCE_DIR"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$ACADOS_SOURCE_DIR/bin:$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
```

Load it from the repository root:

```bash
source .venv/bin/activate
source .envrc
```

Why each variable is needed:

| Variable | Purpose |
| --- | --- |
| `ACADOS_SOURCE_DIR` | lets `acados_template` find headers, libraries, and `t_renderer` |
| `ACADOS_INSTALL_DIR` | acados install location |
| `LD_LIBRARY_PATH` | loads `libacados.so`, `libblasfeo.so`, and `libhpipm.so` |
| `PYTHONPATH` | makes `race_car` and `sensitivity_analysis` importable |
| `PATH` | finds `t_renderer`, `cmake`, and the venv |
| `MPLBACKEND=Agg` | headless plotting (plots are saved to `output/`) |

## 8. Verify

Run the test suite:

```bash
pytest -v
```

Expected: **8 passed** (about 85 seconds on the reference machine).

You can also run the acados OCP self-test directly:

```bash
python race_car/acados_ocp.py
```

It should finish with `Results match`.

## 9. VS Code debugging (F5)

The Python debugger does **not** inherit the environment from `.envrc` or from
a terminal where you activated the venv. In particular, the debugger was
initially launching the system Python 3.14, which does not have the pinned
packages installed.

Two things are required:

1. Force the `.venv` interpreter in the launch configuration.
2. Pass the acados environment variables via the launch configuration's
   `env` block.

The current `.vscode/launch.json` uses this setup (the `pytest` configuration
is the same with `"module": "pytest"`):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File (project env)",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "python": "${workspaceFolder}/.venv/bin/python",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "ACADOS_SOURCE_DIR": "${workspaceFolder}/external/leap-c/external/acados",
        "ACADOS_INSTALL_DIR": "${workspaceFolder}/external/leap-c/external/acados",
        "LD_LIBRARY_PATH": "${workspaceFolder}/external/leap-c/external/acados/lib",
        "PATH": "${env:HOME}/.local/bin:${workspaceFolder}/external/leap-c/external/acados/bin:${env:PATH}",
        "MPLBACKEND": "Agg"
      }
    }
  ]
}
```

Relevant `.vscode/settings.json` entries:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}",
    "ACADOS_SOURCE_DIR": "${workspaceFolder}/external/leap-c/external/acados",
    "ACADOS_INSTALL_DIR": "${workspaceFolder}/external/leap-c/external/acados",
    "LD_LIBRARY_PATH": "${workspaceFolder}/external/leap-c/external/acados/lib",
    "PATH": "${env:HOME}/.local/bin:${workspaceFolder}/external/leap-c/external/acados/bin:${env:PATH}",
    "MPLBACKEND": "Agg"
  }
}
```

After changing these files, reload the VS Code window and select
`.venv/bin/python` via `Python: Select Interpreter`.

The VS Code configuration is machine-specific and intentionally untracked.
`.gitignore` ignores the whole `.vscode/` directory:

```gitignore
.vscode/
```

## 10. Rebuilding / recreating

Rebuild acados after changing the submodule:

```bash
export ACADOS_SOURCE_DIR="$PWD/external/leap-c/external/acados"
export ACADOS_INSTALL_DIR="$ACADOS_SOURCE_DIR"
cmake --build "$ACADOS_SOURCE_DIR/build" --target install -- -j"$(nproc)"
```

Recreate only the Python environment:

```bash
uv venv --clear --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match -r requirements.txt
uv pip install --python .venv/bin/python --no-deps \
  -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
uv pip install --python .venv/bin/python --no-deps -e external/leap-c
```

## Notes

- PyTorch is CPU-only (`torch==2.10.0+cpu`) because the environment is
  CPU-only.
- No system Python packages were modified; Python 3.12, `uv`, CMake, and Ninja
  all live under `~/.local` / `~/.local/share/uv`.
- `external/leap-c/external/acados/build`, `lib`, and `bin` are build outputs
  inside the submodule, not files tracked by this repository.
