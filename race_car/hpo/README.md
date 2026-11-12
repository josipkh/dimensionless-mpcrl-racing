# Hyperparameter optimization with Optuna

This folder contains an Optuna-based HPO workflow for the SAC-ZOP experiments
in [`race_car/scripts/run_sac_zop.py`](../scripts/run_sac_zop.py).  It follows
the RL-HPO recommendations in Eimer, Lindauer & Raileanu, *Hyperparameters in
Reinforcement Learning and How To Tune Them* (ICML 2023) and Henderson et al.,
*Deep Reinforcement Learning that Matters* (AAAI 2018):

- optimize a broad search space with Bayesian optimization (TPE) rather than
  hand-tuned grids;
- use multi-fidelity pruning (Hyperband) so bad configurations are stopped
  early;
- average several validation episodes per checkpoint to reduce noise;
- separate tuning seeds from evaluation/test seeds;
- record the search space, budget, seeds, and best configuration with the
  study.

## Install

Optuna and its dependencies are part of [`requirements.txt`](../../requirements.txt).
If you use an existing environment, install them with:

```bash
python -m pip install optuna plotly
```

## Quick start

Run from the repository root (the same place you normally start
`race_car/scripts/run_sac_zop.py`):

```bash
# 4 parallel trials on an 8-core machine; small 4-parameter search space
python -m race_car.hpo.tune --size default --dimensionless \
    --search-space small --n-trials 12 --n-jobs 4

# broader 9-parameter space (learning rate, gamma, tau, batch size,
# update frequency, learning starts, initial entropy, critic count/width,
# actor width)
python -m race_car.hpo.tune --size default --dimensionless \
    --search-space full --n-trials 24 --n-jobs 4
```

Inspect the study and write plots / the best configuration:

```bash
python -m race_car.hpo.report --study-dir output/hpo/<study_name>
```

Then evaluate the best configuration on **held-out seeds**:

```bash
python race_car/scripts/run_sac_zop.py --size default --dimensionless \
    --seed 5 --output_path output/2026_11_20/12_00_00_sac_zop/tuned \
    --config_overrides output/hpo/<study_name>/best_overrides.yaml
```

To run the same thing through a shell wrapper:

```bash
race_car/scripts/run_hpo_sac_zop.sh --size default --dimensionless --n-trials 24 --n-jobs 4
```

## What is optimized

The objective is the validation score returned by the trainer:

- `--objective best` (default) maximises the best validation checkpoint;
- `--objective final` maximises the last validation checkpoint.

Every trial runs for at most `--train-steps` environment steps.  Hyperband
receives validation scores from the trial's `val_log.csv` and may terminate a
trial at its first rung (`--min-resource`, default 5000 steps) or later rungs.
The default validation setup is `--val-freq 2500` and `--val-num-rollouts 3`,
i.e. three deterministic episodes are averaged at each checkpoint.

The search spaces live in [`search_spaces/`](search_spaces):

| Preset  | Parameters |
| ------- | ---------- |
| `small` | shared learning rate, `gamma`, `tau`, `batch_size` |
| `full`  | the above plus `update_freq`, `train_start`, `init_alpha`, `num_critics`, critic width, actor width |

A search space is a YAML mapping from Optuna parameter names to distribution
specs:

```yaml
lr:
  type: loguniform
  low: 1.0e-6
  high: 1.0e-2
  targets: [lr_q, lr_pi, lr_alpha]   # write one sampled value to several cfg keys
gamma:
  type: uniform
  low: 0.90
  high: 0.9999
batch_size:
  type: categorical
  choices: [32, 64, 128, 256]
```

Supported types are `uniform`, `loguniform`, `int` (optional `step`/`log`) and
`categorical`.  Keys are dotted paths into `SacZopTrainerConfig`; list-valued
choices such as `critic_mlp.hidden_dims` are supported.  To use a custom space,
pass its path, e.g. `--search-space my_space.yaml`.

Fixed settings that apply to every trial can be supplied with
`--base-overrides overrides.yaml` (same dotted-key format).  The driver always
sets validation/logging options that are required for a comparable HPO run
(`val_report_score`, `val_freq`, `val_num_rollouts`, no checkpointing, no
TensorBoard, no video).

## Where files go

A study name defaults to `sac_zop_<size>_<formulation>_<timestamp>`.

```
output/hpo/<study>/          # important, small metadata
├── study.db                 # Optuna SQLite database (resume/analyse here)
├── study_meta.yaml          # creation time, storage, search-space file
├── run_settings.yaml        # budget, seeds, objective, fixed overrides
├── search_space.yaml        # exact resolved search space
├── invocations.yaml         # every `tune.py` invocation (resume history)
├── best_overrides.yaml      # ready for --config_overrides
├── best_trial.yaml          # best trial metadata
├── trials.csv               # all trials, flattened
├── report.txt               # summary from report.py
└── optimization_history.html, param_importances.html, parallel_coordinate.html

/tmp/hpo/<study>/            # large per-trial artifacts (safe to delete)
└── trial_0000/
    ├── config_overrides.yaml
    ├── sampled_params.yaml
    ├── config.yaml
    ├── val_log.csv          # streamed to Optuna for pruning
    ├── output.log
    └── hpo_subprocess.log
```

Use `--keep-best-artifacts` to copy the best trial's small logs into
`output/hpo/<study>/best_trial`, and `--cleanup-artifacts` to delete the other
`/tmp` trial folders after the study finishes.

## Resuming and parallel trials

The SQLite storage makes studies resumable and safe to inspect while running:

```bash
python -m race_car.hpo.tune --size default --dimensionless \
    --study-name sac_zop_default_dimensionless_full \
    --search-space full --n-trials 24 --n-jobs 4 --load-if-exists
```

`--n-jobs` only controls how many **training subprocesses** run concurrently;
Optuna itself stays in one process.  With 8 CPU cores, `--n-jobs 4` is a good
starting point (each run uses roughly 1.5 cores).  You can also use an external
Optuna storage:

```bash
--storage sqlite:////absolute/path/to/study.db
# or a database server URL supported by Optuna
```

## Budget guidance

A useful default is 20-30 trials with `--train-steps 50000`,
`--min-resource 5000`, and `--reduction-factor 3`.  Hyperband assigns trials to
successively larger brackets, so most runs are stopped after 5k-15k steps.  For
a production-quality comparison, increase both the budget and the number of
tuning seeds, then re-evaluate the incumbent on separate test seeds (the
literature recommends e.g. tune on seeds 0-4, test on seeds 5-14).  Record the
tuning seeds and budget in the paper/report; the study metadata stores them for
you.

Runtime reference on the 8-core development machine (CPU, `float64`, default
car): about 2 minutes for a 1000-step run including acados compilation, i.e.
roughly one minute per 1000 training steps after compilation.  Compilation is
repeated per trial because each trial is a separate process.

## Testing the plumbing without training

```bash
# print sampled configurations, launch nothing
python -m race_car.hpo.tune --search-space full --n-trials 3 --dry-run

# run the small self-tests
python race_car/hpo/config.py
python race_car/hpo/search_space.py
python race_car/hpo/runner.py
```

## Extending to MPC / controller parameters

The current workflow tunes the **RL trainer** hyperparameters.  The default MPC
weights live in `race_car/utils/config.py` (`CarParams`) rather than in
`SacZopTrainerConfig`, so they are not exposed as dotted overrides here.  If
you want to tune the controller's default cost weights as well, the clean
extension is to add a small wrapper config (or a `--base-overrides`-style hook
in `run_sac_zop.py`) that maps sampled values onto `CarParams` before the
controller is constructed, then add a separate search-space YAML.  Keep RL-HPO
and MPC-weight tuning as separate studies so their budgets and seeds are
comparable.
