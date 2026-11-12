"""Optuna-based hyperparameter optimization for the race car SAC-ZOP experiments.

The package is intentionally lightweight at import time: :mod:`optuna` is only
imported by the modules that actually need it (``tune`` and ``report``).  This
keeps ``race_car.scripts.run_sac_zop`` usable without Optuna installed.
"""
