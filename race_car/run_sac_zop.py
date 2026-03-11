"""Main script to run SAC-ZOP experiments."""

from dataclasses import asdict
from typing import Literal

from leap_c.run import default_name, default_output_path, init_run
from leap_c.torch.rl.sac_zop import SacZopTrainer, SacZopTrainerConfig

from race_car.env import RaceCarEnv
from race_car.controller import RaceCarController


def create_cfg(
    seed: int,
    ckpt_modus: Literal["best", "last", "all", "none"] = "last",
) -> SacZopTrainerConfig:
    # ---- Configuration ----
    cfg = SacZopTrainerConfig()

    # ---- Section: cfg.trainer ----
    cfg.seed = seed
    cfg.train_steps = 50_000
    cfg.train_start = 0
    cfg.val_freq = 10_000
    cfg.val_num_rollouts = 1
    cfg.val_deterministic = True
    cfg.val_num_render_rollouts = 1
    cfg.val_render_mode = "rgb_array"
    cfg.val_report_score = "cum"
    cfg.ckpt_modus = ckpt_modus
    cfg.batch_size = 64
    cfg.buffer_size = 500_000
    cfg.gamma = 0.99
    cfg.tau = 0.005
    cfg.soft_update_freq = 1
    cfg.lr_q = 0.001
    cfg.lr_pi = 0.001
    cfg.lr_alpha = 0.001
    cfg.init_alpha = 0.02
    cfg.target_entropy = None
    cfg.entropy_reward_bonus = True
    cfg.num_critics = 2
    cfg.update_freq = 4

    # ---- Section: cfg.log ----
    cfg.log.verbose = True
    cfg.log.interval = 1000
    cfg.log.window = 10_000
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = False
    cfg.log.wandb_logger = False
    cfg.log.wandb_init_kwargs = {}

    # ---- Section: cfg.critic_mlp ----
    cfg.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.critic_mlp.activation = "relu"
    cfg.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.actor ----
    cfg.actor.noise = "param"
    cfg.actor.extractor_name = "identity"
    cfg.actor.distribution_name = "squashed_gaussian"
    cfg.actor.residual = True  # important to start from default param values
    cfg.actor.entropy_correction = False

    # ---- Section: cfg.actor.mlp ----
    # cfg.actor.mlp.hidden_dims = (256, 256, 256)
    cfg.actor.mlp.hidden_dims = None
    cfg.actor.mlp.activation = "relu"
    cfg.actor.mlp.weight_init = "orthogonal"

    return cfg


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params, get_default_car_params_tracking
    from race_car.utils.scaling import get_large_car_params

    dimensionless = False
    seed = 0
    use_wandb = False

    output_path = default_output_path(seed, tags=["sac_zop", "racing"])
    cfg = create_cfg(seed=seed)

    car_params = get_default_car_params()
    train_env = RaceCarEnv(car_params=car_params, dimensionless=dimensionless, mpc_car_params=car_params, render_mode=None)
    val_env = RaceCarEnv(car_params=car_params, dimensionless=dimensionless, mpc_car_params=car_params, render_mode=cfg.val_render_mode)
    controller = RaceCarController(car_params=car_params, dimensionless=dimensionless)

    if use_wandb:
        config_dict = asdict(cfg)
        cfg.log.wandb_logger = True
        cfg.log.wandb_init_kwargs = {
            "entity": None,
            "project": "leap-c",
            "name": default_name(seed, tags=["sac_zop", "racing"]),
            "config": config_dict,
        }

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

