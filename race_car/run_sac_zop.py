"""Main script to run SAC-ZOP experiments."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from leap_c.examples import ExampleControllerName, ExampleEnvName, create_controller, create_env
from leap_c.run import default_controller_code_path, default_name, default_output_path, init_run
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig

from race_car.env import RaceCarEnv
from race_car.controller import RaceCarController


@dataclass
class RunSacZopConfig:
    """Configuration for running SAC-ZOP experiments."""

    env: RaceCarEnv
    controller: RaceCarController
    trainer: SacZopTrainerConfig = field(default_factory=SacZopTrainerConfig)
    extractor: ExtractorName = "identity"


def create_cfg(
    env: RaceCarEnv,
    controller: RaceCarController,
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> RunSacZopConfig:
    # ---- Configuration ----
    cfg = RunSacZopConfig(env=env, controller=controller)

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = seed
    cfg.trainer.train_steps = 200_000
    cfg.trainer.train_start = 0
    cfg.trainer.val_freq = 10_000
    cfg.trainer.val_num_rollouts = 1
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 1
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = ckpt_modus
    cfg.trainer.batch_size = 64
    cfg.trainer.buffer_size = 1_000_000
    cfg.trainer.gamma = 0.99
    cfg.trainer.tau = 0.005
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.lr_q = 0.001
    cfg.trainer.lr_pi = 0.001
    cfg.trainer.lr_alpha = 0.001
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 2
    cfg.trainer.update_freq = 4

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10_000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = False
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor ----
    cfg.trainer.actor.noise = "param"
    cfg.trainer.actor.extractor_name = cfg.extractor
    cfg.trainer.actor.distribution_name = "squashed_gaussian"
    cfg.trainer.actor.residual = True
    cfg.trainer.actor.entropy_correction = False

    # ---- Section: cfg.trainer.actor.mlp ----
    # cfg.trainer.actor.mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor.mlp.hidden_dims = None
    cfg.trainer.actor.mlp.activation = "relu"
    cfg.trainer.actor.mlp.weight_init = "orthogonal"

    return cfg


def run_sac_zop(
    cfg: RunSacZopConfig,
    output_path: str | Path,
    device: str = "cuda",
    with_val: bool = False,
) -> float:
    """Run the SAC-ZOP trainer.

    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
        reuse_code_dir: The directory to reuse compiled code from, if any.
        with_val: Whether to use a validation environment.
    """
    val_env = cfg.env if with_val else None
    trainer = SacZopTrainer(
        cfg=cfg.trainer,
        val_env=val_env,
        output_path=output_path,
        device=device,
        train_env=cfg.env,
        controller=cfg.controller,
        extractor_cls=cfg.extractor,
    )
    init_run(trainer, cfg, output_path)
    return trainer.run()


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params, get_default_car_params_tracking
    from race_car.utils.scaling import get_large_car_params

    dimensionless = False
    output_path = default_output_path(seed=0, tags=["sac_zop"])
    render_mode = "rgb_array"
    seed = 0
    use_wandb = False

    car_params = get_default_car_params_tracking()
    env = RaceCarEnv(car_params=car_params, dimensionless=dimensionless, mpc_car_params=car_params, render_mode=render_mode)
    controller = RaceCarController(car_params=car_params, dimensionless=dimensionless)

    cfg = create_cfg(env=env, controller=controller, seed=seed)

    if use_wandb:
        config_dict = asdict(cfg)
        cfg.trainer.log.wandb_logger = True
        cfg.trainer.log.wandb_init_kwargs = {
            "entity": None,
            "project": "leap-c",
            "name": default_name(seed, tags=["sac_zop", "racing"]),
            "config": config_dict,
        }

    run_sac_zop(
        cfg=cfg,
        output_path=output_path,
        device="cpu",
        with_val=True,
    )

    # parser = ArgumentParser()
    # parser.add_argument("--output_path", type=Path, default=None)
    # parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--env", type=str, default="cartpole")
    # parser.add_argument("--controller", type=str, default=None)
    # parser.add_argument("--with-val", action="store_true", help="Enable validation environment")
    # parser.add_argument(
    #     "--ckpt-modus",
    #     type=str,
    #     default=None,
    #     choices=["none", "last", "all", "best"],
    #     help="Checkpoint mode. Defaults to 'best' with --with-val, 'last' otherwise.",
    # )
    # parser.add_argument(
    #     "-r",
    #     "--reuse_code",
    #     action="store_true",
    #     help="Reuse compiled code. The first time this is run, it will compile the code.",
    # )
    # parser.add_argument("--reuse_code_dir", type=Path, default=None)
    # parser.add_argument("--use-wandb", action="store_true")
    # parser.add_argument("--wandb-entity", type=str, default=None)
    # parser.add_argument("--wandb-project", type=str, default="leap-c")
    # args = parser.parse_args()

    # if args.ckpt_modus is not None:
    #     ckpt_modus = args.ckpt_modus
    # elif args.with_val:
    #     ckpt_modus = "best"
    # else:
    #     ckpt_modus = "last"

    # cfg = create_cfg(args.env, args.controller, args.seed, ckpt_modus)



    # if args.output_path is None:
    #     output_path = default_output_path(
    #         seed=args.seed, tags=["sac_zop", args.env, args.controller]
    #     )
    # else:
    #     output_path = args.output_path

    # if args.reuse_code and args.reuse_code_dir is None:
    #     reuse_code_dir = default_controller_code_path()
    # elif args.reuse_code_dir is not None:
    #     reuse_code_dir = args.reuse_code_dir
    # else:
    #     reuse_code_dir = None

    # run_sac_zop(
    #     cfg=cfg,
    #     output_path=output_path,
    #     device=args.device,
    #     reuse_code_dir=reuse_code_dir,
    #     with_val=args.with_val,
    # )
