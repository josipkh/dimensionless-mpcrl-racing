"""Script to run SAC-ZOP (MPC is effectively part of the environment) with the race car example."""

from pathlib import Path
import time
from datetime import datetime, timedelta
from argparse import ArgumentParser
from tempfile import mkdtemp

from leap_c.run import default_output_path, init_run
from leap_c.torch.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig
from leap_c.utils.git import log_git_hash_and_diff

import race_car
from race_car.env import RaceCarEnv
from race_car.controller import RaceCarController
from race_car.utils.log_utils import tee_logging
from race_car.utils.config import get_default_car_params_tracking, get_large_car_params_tracking


def create_cfg(seed: int, train_steps: int) -> SacZopTrainerConfig:
    # ---- Configuration ----
    cfg = SacZopTrainerConfig()

    # ---- Section: cfg.trainer ----
    cfg.seed = seed
    cfg.train_steps = train_steps
    cfg.train_start = 1_000
    cfg.val_freq = 2_000
    cfg.val_num_rollouts = 1
    cfg.val_deterministic = True
    cfg.val_num_render_rollouts = 1
    cfg.val_render_mode = "rgb_array"
    cfg.val_report_score = "cum"
    cfg.ckpt_modus = "last"
    cfg.batch_size = 64
    cfg.buffer_size = 500_000
    cfg.gamma = 0.99
    cfg.tau = 0.005
    cfg.soft_update_freq = 1
    cfg.lr_q = 3e-4
    cfg.lr_pi = 3e-4
    cfg.lr_alpha = 3e-4
    cfg.init_alpha = 0.005
    cfg.target_entropy = None
    cfg.entropy_reward_bonus = True
    cfg.num_critics = 2
    cfg.update_freq = 4

    # ---- Section: cfg.log ----
    cfg.log.verbose = True
    cfg.log.interval = 1_000
    cfg.log.window = 10_000
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = True
    cfg.log.wandb_logger = False
    cfg.log.wandb_init_kwargs = {}

    # ---- Section: cfg.critic_mlp ----
    cfg.critic_mlp.hidden_dims = (64, 64, 64)
    cfg.critic_mlp.activation = "relu"
    cfg.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.actor ----
    cfg.actor.noise = "param"
    cfg.actor.extractor_name = "scaling"
    cfg.actor.distribution_name = "mode_concentration_beta"  # "squashed_gaussian"
    cfg.actor.residual = True  # important to start from default param values
    cfg.actor.entropy_correction = False

    # ---- Section: cfg.actor.mlp ----
    # cfg.actor.mlp.hidden_dims = (256, 256, 256)
    # cfg.actor.mlp.activation = "relu"
    # cfg.actor.mlp.weight_init = "orthogonal"
    cfg.actor.mlp.hidden_dims = None

    return cfg


if __name__ == "__main__":

    keep_output = False  # only relevant if running this file directly, bash script always saves the output

    parser = ArgumentParser()
    parser.add_argument("--size", type=str, default="default")  # "default" or "large"
    parser.add_argument("--dimensionless", action='store_true')  # defaults to False, call with "--dimensionless" to set to True
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--train_steps", type=int, default=1)
    args = parser.parse_args()

    # select the car parameters
    if args.size == "default":
        from race_car.utils.config import get_default_car_params_tracking
        car_params = get_default_car_params_tracking()
    elif args.size == "large":
        from race_car.utils.config import get_large_car_params_tracking
        car_params = get_large_car_params_tracking()
    else:
        raise ValueError("Please specify the car size with --size (e.g. --size default)")

    # other settings
    tags = ["sac_zop_transfer", args.size, "dimensionless" if args.dimensionless else "dimensional"]
    if args.timestamp:
        dt = datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
        timestamp_formatted = dt.strftime("%Y_%m_%d/%H_%M_%S")
        output_path = f"output/{timestamp_formatted}_sac_zop_transfer/{'dimensionless' if args.dimensionless else 'dimensional'}/seed_{args.seed}"
        output_path = Path(output_path)
    else:
        output_path = default_output_path(args.seed, tags=tags) if keep_output else Path(mkdtemp())
    output_path.mkdir(parents=True, exist_ok=True)

    log_git_hash_and_diff(output_path / "my_git_diff.txt", Path(race_car.__path__[0]).parent)  # save the git diff for this repo as well

    with tee_logging(output_path):
        t_start = time.monotonic()
        print(f"Running SAC-ZOP with the {args.size} car, dimensionless={args.dimensionless}, seed={args.seed}")
        cfg = create_cfg(seed=args.seed, train_steps=args.train_steps)
        train_env = RaceCarEnv(car_params=car_params, dimensionless=args.dimensionless, mpc_car_params=car_params, render_mode=None)
        val_env = RaceCarEnv(car_params=car_params, dimensionless=args.dimensionless, mpc_car_params=car_params, render_mode=cfg.val_render_mode)
        controller = RaceCarController(car_params=car_params, dimensionless=args.dimensionless)

        trainer = SacZopTrainer(
            cfg=cfg,
            controller=controller,
            train_env=train_env,
            val_env=val_env,
            output_path=output_path,
            device="cpu",
        )

        init_run(trainer, cfg, output_path)
        trainer.run()

        elapsed = time.monotonic() - t_start
        print(f"\nRun finished. Wall-clock time: {timedelta(seconds=round(elapsed))} ({elapsed:.1f} s)")
