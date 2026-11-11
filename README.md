# Dimensionless MPC-based RL for racing

This project applies the methodology of dimensionless (scale-invariant) reinforcement learning (RL) based on model predictive control (MPC) described in [this paper](https://arxiv.org/abs/2512.08667) to a race car example based on [this paper](https://www.sciencedirect.com/science/article/pii/S2405896320317845).

The goal is to drive around the track in minimum time, with a progress-maximizing reformulation. Starting from an MPC cost which favors sticking to the centerline (similar to a careful driver when on a new track), RL should adapt the weights such that time-optimal behaviour is achieved.

Additionally, we analyze the importance of each dimensionless group for:

- dynamic similarity - based on [activity scores](https://www.sciencedirect.com/science/article/pii/S0951832016303052), comparing one-step prediction accuracy
- closed-loop performance - based on section II.A in [this paper](https://ieeexplore.ieee.org/abstract/document/10938338), estimating the impact of plant-model mismatch

Both analytical methods are compared to empirical, finite difference experiments.

The implementation relies on [leap-c](https://leap-c.github.io/leap-c/) and the differentiable MPC functionality in [acados](https://docs.acados.org/). To reproduce the results, please use the versions pinned in the folder `external`:

- `leap-c`: [v0.2.0-alpha](https://github.com/leap-c/leap-c/releases/tag/v0.2.0-alpha) (slightly modified)
- `acados`: [v0.5.5](https://github.com/acados/acados/releases/tag/v0.5.5)

## One-click reproducibility with Docker

From the repository root, run one command:

```bash
./docker/reproduce.sh
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\reproduce.ps1
```

Both commands set up Docker if needed, build the pinned CPU-only image
(Python environment plus acados and leap-c from the submodules), run the test
suite and all sensitivity analyses, and then run every RL experiment. No host
Python, acados, or CMake installation is needed. If Docker is already installed
and running, they run without further prompts.

The RL suites take many hours; see [`docker/README.md`](docker/README.md) for
shorter modes, the Docker setup details, and the cases where a one-time
reboot or admin approval may be required.

## Local development setup

See [SETUP.md](SETUP.md) for the Python 3.12 environment, acados build, shell
environment variables, test commands, and VS Code debugger configuration used
for local development.
