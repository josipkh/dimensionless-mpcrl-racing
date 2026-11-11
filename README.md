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

## Local development setup

See [SETUP.md](SETUP.md) for the Python 3.12 environment, acados build, shell
environment variables, test commands, and VS Code debugger configuration used
for local development.

## Running the transfer experiments

Transfer experiments pretrain SAC-ZOP on one car (the *source*) and fine-tune it
on the other (the *target*). Both runs share one output directory, so the target
run continues from the source checkpoint. Every experiment is repeated for both
MPC formulations (`dimensional` and `dimensionless`) and seeds `0`-`4`.

```bash
# small -> large (default direction)
race_car/scripts/run_all_sac_zop_transfer.sh

# large -> small
TRANSFER_DIRECTION=large_to_default race_car/scripts/run_all_sac_zop_transfer.sh
```

The source car is pretrained for 50k steps and the target car is fine-tuned for
100k steps. The floating point precision is selectable per run and defaults to
`float32`:

```bash
python race_car/scripts/run_sac_zop_transfer.py \
    --size default --dimensionless --seed 0 --train_steps 50000 --dtype float32
```

Average the per-seed TensorBoard logs afterwards with
`race_car/scripts/average_results.sh <results-folder>`.

### Reproducing the 2026_06_12 / 2026_06_13 runs

`output/2026_06_12` (small -> large) and `output/2026_06_13` (large -> small) are
the reference transfer runs. Their script settings are matched by the current
scripts: the wide parameter bounds in `race_car/utils/config.py`, the 50k/100k
source/target budgets, and `--lax-steps` in `average_results.sh`.

The `leap-c` revision used for those runs (`5c7fe3c`, plus a local
`save_p_global=True` patch) has since been rewritten and is no longer reachable.
The currently pinned `leap-c` (`c8a7e3e`) differs in ways that affect results:

- min-max normalisation is applied to `[-1, 1]` (`f982313`) instead of the
  `[0, 1]` used in June;
- the local `save_p_global=True` patch in `create_batch_solver` is absent;
- `acados` is at v0.5.5.
