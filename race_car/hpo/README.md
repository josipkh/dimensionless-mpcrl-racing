# Hyperparameter optimization with Optuna

Minimal HPO for the SAC-ZOP experiments in
[`race_car/scripts/run_sac_zop.py`](../scripts/run_sac_zop.py).  It tunes only
five trainer hyperparameters and discards runs whose MPC solver becomes
unreliable.

## What is tuned

| Parameter | Distribution | Range |
| --------- | ------------ | ----- |
| `tau` | log-uniform | 1e-4 – 1e-1 |
| `lr_q` | log-uniform | 1e-5 – 1e-2 |
| `lr_pi` | log-uniform | 1e-5 – 1e-2 |
| `lr_alpha` | log-uniform | 1e-5 – 1e-2 |
| `init_alpha` | log-uniform | 1e-4 – 1e-1 |

The ranges are centred on the June 2026 baseline (`tau=0.005`,
`lr_q=lr_pi=lr_alpha=3e-4`, `init_alpha=0.005`) and span about one decade in
each direction.  Edit `SEARCH_RANGES` in [`tune.py`](tune.py) to change them.

Everything else is fixed by `create_cfg` in `run_sac_zop.py`:

- fixed default MPC parameters (`CarParams`) — the policy learns the
  parameters directly, with no hidden layers (`actor.mlp.hidden_dims = None`,
  residual learning from the defaults);
- critic network `[64, 64, 64]`, batch size 64, `gamma=0.99`,
  `update_freq=4`, `train_start=1000`;
- `mode_concentration_beta` parameter noise with the `scaling` extractor.

## Reliability filter

Each validation writes the MPC solve statistics to `val_policy_log.csv`.  If
any validation checkpoint reports `success_rate < 0.85`, the trial is stopped
and marked pruned (it cannot become the best trial).  The threshold is
`--min-solve-rate`, and the minimum solve rate of completed trials is stored in
the study and in `trials.csv`.

The objective is the **best validation score** among checkpoints that passed
the reliability filter.  Median pruning additionally stops clearly bad trials
early.

## Quick start

Run from the repository root:

```bash
# 20 trials, 4 in parallel, 50k steps each
python -m race_car.hpo.tune --size default --dimensionless --n-trials 20 --n-jobs 4
```

Or through the shell wrapper:

```bash
race_car/scripts/run_hpo_sac_zop.sh --size default --dimensionless --n-trials 20 --n-jobs 4
```

Useful options:

```bash
--train-steps 50000        # per-trial budget
--val-freq 2500            # validation interval (and solve-rate check)
--val-num-rollouts 3       # average validation over several episodes
--min-solve-rate 0.85      # reliability threshold
--tuning-seeds 0 1 2       # seeds cycled over trials
--trial-timeout-min 90     # kill a runaway trial
--dry-run                  # print sampled configurations only
```

Evaluate the winner on **held-out seeds** (the tuning seeds should not be used
for the final comparison):

```bash
python race_car/scripts/run_sac_zop.py --size default --dimensionless \
    --seed 5 --output_path output/2026_11_20/12_00_00_sac_zop/tuned \
    --config_overrides output/hpo/<study>/best_overrides.yaml
```

## Outputs

```
output/hpo/<study>/          # small, important files
├── study.db                 # Optuna database (resume/inspect)
├── run_settings.yaml        # arguments, search ranges, fixed overrides
├── best_overrides.yaml      # ready for --config_overrides
├── trials.csv               # one row per trial, incl. solve rate and discard reason
└── best_trial/              # only with --keep-best
    ├── val_log.csv
    ├── val_policy_log.csv
    ├── config.yaml
    └── output.log

/tmp/hpo/<study>/trial_XXXX/ # large per-trial artifacts (safe to delete)
```

Resume an interrupted study:

```bash
python -m race_car.hpo.tune --study-name hpo_default_dimensionless_20261112_120000 \
    --n-trials 20 --n-jobs 4 --load-if-exists
```

`--n-jobs` controls how many training subprocesses run at once; on an 8-core
machine, 4 is a good default.  Use `--cleanup-artifacts` to delete non-best
trial directories when the study finishes.

## Notes from the June 2026 runs

The June experiments used the fixed configuration above for 100k steps.  Best
validation scores were roughly -28 to -31 (dimensional) and -60 to -79
(dimensionless).  Solve rates were typically 0.84–1.0, but many runs had
transient dips below 0.85 during training; this is why the reliability filter
examines every validation checkpoint rather than only the final one.

## Testing

```bash
python race_car/hpo/overrides.py
PYTHONPATH=. python race_car/hpo/tune.py
python -m pytest tests/test_main_blocks.py -k hpo -q
```

`tune.py`'s self-test runs the subprocess/log-streaming code on a fake trainer
and checks both the normal path and the `DiscardedTrial` path.
