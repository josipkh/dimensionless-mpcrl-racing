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

The direction is recorded automatically in a `transfer.json` manifest at the
root of each batch. It lists the direction(s), source/target cars, step budgets,
formulations, seeds, timestamp and git revision, so results no longer need to be
matched to a direction by hand.

By default both directions run in one batch. The batch folder is
direction-neutral and holds one subfolder per direction, each with its own
manifest, plus a root manifest listing both directions:

```
output/<date>/<time>_sac_zop_transfer/
├── transfer.json
├── small_to_large/{dimensional,dimensionless}/seed_0..4
└── large_to_small/{dimensional,dimensionless}/seed_0..4
```

```bash
# both directions, one batch folder (default)
race_car/scripts/run_all_sac_zop_transfer.sh

# small -> large only
TRANSFER_DIRECTION=small_to_large race_car/scripts/run_all_sac_zop_transfer.sh

# large -> small only
TRANSFER_DIRECTION=large_to_small race_car/scripts/run_all_sac_zop_transfer.sh
```

Explicit single-direction batches keep the old flat layout with a direction-tagged
folder name (`<time>_sac_zop_transfer_<direction>`).

The source car is pretrained for 50k steps and the target car is fine-tuned for
100k steps. The floating point precision is selectable per run and defaults to
`float32`:

```bash
python race_car/scripts/run_sac_zop_transfer.py \
    --size small --direction small_to_large --phase source \
    --dimensionless --seed 0 --train_steps 50000 --dtype float32
```

`--direction` selects the car pair (`small_to_large` or `large_to_small`)
and `--phase` says which half of the chain is running (`source` or `target`).
The script checks that `--size` agrees with both; a mismatch is rejected.
Note that a single manual run does not write `transfer.json` — the batch
launcher does that once per experiment. The launcher also passes `--run_root`
to both phases of a chain so they share one folder, including the per-direction
subfolder of the default both-direction batch; the example above falls back to
the timestamp-based path when `--run_root` is omitted.

Average the per-seed TensorBoard logs afterwards with
`race_car/scripts/average_results.sh <results-folder>`. A batch root can be
passed directly — the script recurses into its direction subfolders. This is
also done automatically: each `run_all_*` script launches an extra `<timestamp>_averager`
tmux session that waits for the whole batch to finish and then averages the
results. The seeds of `dimensional` and `dimensionless` are always averaged
separately, and every timestamp folder under the given results folder is handled
(so passing a date folder averages each of its timestamps on its own). Set
`AUTO_AVERAGE=0` to skip it, or run the watcher by hand for an in-flight batch:

```bash
race_car/scripts/wait_and_average.sh <timestamp> <results-folder>
```

If a reduction fails, `tb-reducer`'s output is saved next to the folders it
would have written, as `<formulation>/AVERAGE_FAILED.log`, the watcher exits
non-zero, and its tmux session is left open instead of closing itself. This
makes a failed averaging easy to spot and to inspect. After a successful
reduction the watcher also writes `plot_transfer.pdf` into each transfer batch
folder; non-transfer batches are skipped.

### Plotting transfer runs

The plotting logic lives in `race_car/utils/plotting.py` and is run automatically
by the watcher after averaging. It reads the labels from the manifests and plots
all four combinations (2 directions × 2 formulations): colour marks the
direction, linestyle the formulation. Older runs without a manifest are inferred
from the first line of their `output.log`.

To regenerate a plot by hand, pass a batch folder (or a date folder holding
several batches). One PDF is saved per batch folder:

```bash
python -m race_car.utils.plotting \
    output/2026_11_20/13_58_16_sac_zop_transfer
```

Two separate run folders can still be passed for older experiments that stored
each direction in its own folder:

```bash
python -m race_car.utils.plotting \
    output/2026_11_12/10_19_21_sac_zop_transfer \
    output/2026_11_12/10_19_23_sac_zop_transfer \
    --output output/legacy_transfer.pdf
```

With no folder argument, the module instead runs the local smoke test that loads
`race_car/jkh_test/small_car_results.npz` and renders the result plots.

### Reproducing the 2026_06_12 / 2026_06_13 runs

`output/2026_06_12` (small -> large) and `output/2026_06_13` (large -> small) are
the reference transfer runs. They predate the direction-tagged folder names, but
their direction is still recoverable from the first line of any `output.log`
(that is the legacy fallback in `race_car/utils/plotting.py`). Their script settings
are matched by the current scripts: the wide parameter bounds in
`race_car/utils/config.py`, the 50k/100k source/target budgets, and `--lax-steps`
in `average_results.sh`.

The `leap-c` revision used for those runs (`5c7fe3c`) has since been rewritten
and is no longer reachable. The currently pinned `leap-c` (`c8a7e3e`) matches
those runs on min-max normalisation (applied to `[-1, 1]`) and is trained in
`float32`. The local `save_p_global=True` flag is only used for debugging and
does not affect the results, so it is not restored. The remaining known
difference is that `acados` is at v0.5.5.
