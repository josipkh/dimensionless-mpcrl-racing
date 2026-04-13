"""Main script to run controllers with default parameters."""

from pathlib import Path
from typing import Any, Generator
from tempfile import mkdtemp
from argparse import ArgumentParser
from datetime import datetime, timedelta
import time

import gymnasium as gym
import torch
from numpy import ndarray

from leap_c.controller import CtxType, ParameterizedController
from leap_c.run import default_output_path, init_run
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.trainer import Trainer, TrainerConfig

from race_car.env import RaceCarEnv
from race_car.controller import RaceCarController
from race_car.utils.log_utils import tee_logging


class BaselineTrainer(Trainer):

    def __init__(
        self,
        cfg: TrainerConfig,
        eval_env: gym.Env,
        output_path: str | Path,
        device: str,
        controller: ParameterizedController[CtxType],
    ) -> None:
        
        super().__init__(cfg, eval_env, output_path, device)
        self.controller = controller

        buffer = ReplayBuffer(1, device, collate_fn_map=controller.collate_fn_map)  # type: ignore
        self.collate_fn = buffer.collate

    def train_loop(self) -> Generator[tuple[int, float], None, None]:
        # validation-only mode: just yield dummy values
        while True:
            yield 1, 0.0

    def act(
        self, obs: ndarray, deterministic: bool = False, state: Any | None = None
    ) -> tuple[ndarray, Any, dict[str, float] | None]:

        obs_batched = self.collate_fn([obs])
        default_param = self.controller.default_param(obs_batched)
        # default_param = torch.from_numpy(default_param).to(self.device)
        default_param = torch.from_numpy(default_param.copy()).to(self.device)  # TODO: check this
        ctx, action = self.controller(obs_batched, default_param, ctx=state)        
        action = action.cpu().numpy()[0]
        return action, ctx, ctx.log


def create_cfg(seed: int) -> TrainerConfig:

    cfg = TrainerConfig()

    # ---- Section: cfg.trainer ----
    cfg.seed = seed
    cfg.train_start = 0
    cfg.val_num_rollouts = 1
    cfg.val_deterministic = True
    cfg.val_num_render_rollouts = 1
    cfg.val_render_mode = render_mode
    cfg.val_report_score = "cum"
    cfg.ckpt_modus = "none"
    cfg.train_steps = 0
    cfg.val_freq = 1

    # ---- Section: cfg.log ----
    cfg.log.verbose = True
    cfg.log.interval = 1
    cfg.log.window = 1
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = False
    cfg.log.wandb_logger = False
    cfg.log.wandb_init_kwargs = {}

    return cfg


if __name__ == "__main__":
    
    seed = 0
    render_mode = "rgb_array"
    keep_output = False  # only relevant if running this file directly

    parser = ArgumentParser()
    parser.add_argument("--size", type=str, default="default")  # "default" or "large"
    parser.add_argument("--dimensionless", action='store_true')  # defaults to False, call with "--dimensionless" to set to True
    parser.add_argument("--timestamp", type=str, default=None)
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
    
    # car_params = get_default_car_params_tracking()
    # args.dimensionless = True
    
    tags = ["baseline", args.size, "dimensionless" if args.dimensionless else "dimensional"]
    if args.timestamp:
        dt = datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
        timestamp_formatted = dt.strftime("%Y_%m_%d/%H_%M_%S")
        output_path = f"output/{timestamp_formatted}_baseline_seed_{seed}/{args.size}_{'dimensionless' if args.dimensionless else 'dimensional'}"
    else:
        output_path = default_output_path(seed, tags=tags) if keep_output else str(Path(mkdtemp()))
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with tee_logging(Path(output_path)):
        t_start = time.monotonic()
        try:
            print(f"Running baseline with the {args.size} car, dimensionless={args.dimensionless}, seed={seed}")
            env = RaceCarEnv(car_params=car_params, dimensionless=args.dimensionless, mpc_car_params=car_params, render_mode=render_mode)
            controller = RaceCarController(car_params=car_params, dimensionless=args.dimensionless)

            cfg = create_cfg(seed=seed)

            trainer = BaselineTrainer(
                cfg=cfg,
                eval_env=env,
                controller=controller,
                output_path=output_path,
                device="cpu",        
            )

            init_run(trainer, cfg, output_path)
            trainer.run()

            elapsed = time.monotonic() - t_start
            print(f"\nRun finished. Wall-clock time: {timedelta(seconds=round(elapsed))} ({elapsed:.1f} s)")

        except Exception as e:
            import traceback
            print(f"Run crashed with: {e}")
            traceback.print_exc()
            raise