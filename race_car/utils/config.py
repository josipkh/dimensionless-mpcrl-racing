"""Defines the parameters of the race car OCP and their default values."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from leap_c.ocp.acados.parameters import AcadosParameter


@dataclass(kw_only=True)
class CarParams:
    # Chassis parameters
    m: np.ndarray         # [kg] mass of the car
    l: np.ndarray         # [m] length of the car
    lr: np.ndarray        # [m] distance from CG to rear axle

    # Longitudinal force parameters (eq. 4)
    cm1: np.ndarray       # [kgm/s^2] 
    cm2: np.ndarray       # [kg/s]
    cr0: np.ndarray       # [kgm/s^2]
    cr2: np.ndarray       # [kg/m]
    cr3: np.ndarray       # [s/m]

    # Cost weight matrices, defined by the square roots of their diagonal entries
    q_diag_sqrt: np.ndarray     # stage cost, state residuals
    r_diag_sqrt: np.ndarray     # stage cost, input residuals
    qe_diag_sqrt: np.ndarray    # terminal cost, state residuals

    # Slack cost weights
    slack_n_linear: np.ndarray          # linear cost on the position constraint [1/m]
    slack_n_quadratic: np.ndarray       # quadratic cost on the position constraint [1/m^2]
    slack_acc_linear: np.ndarray        # linear cost on the acceleration constraint [1/(m/s^2)]
    slack_acc_quadratic: np.ndarray     # quadratic cost on the acceleration constraint [1/(m/s^2)^2]

    # Constraints
    a_lat_max: np.ndarray   # maximum lateral acceleration [m/s^2]
    a_lat_min: np.ndarray   # minimum lateral acceleration [m/s^2]
    a_long_max: np.ndarray  # maximum longitudinal acceleration [m/s^2]
    a_long_min: np.ndarray  # minimum longitudinal acceleration [m/s^2]
    n_max: np.ndarray       # maximum lateral deviation from center line [m]
    n_min: np.ndarray       # minimum lateral deviation from center line [m]

    D_max: np.ndarray       # maximum duty cycle [-]
    D_min: np.ndarray       # minimum duty cycle [-]
    delta_max: np.ndarray   # maximum steering angle [rad]
    delta_min: np.ndarray   # minimum steering angle [rad]
    dD_max: np.ndarray      # maximum duty cycle rate [1/s]
    dD_min: np.ndarray      # minimum duty cycle rate [1/s]
    ddelta_max: np.ndarray  # maximum steering angle rate [rad/s]
    ddelta_min: np.ndarray  # minimum steering angle rate [rad/s]

    # Controller parameters
    dt: np.ndarray          # time step [s]
    gamma: np.ndarray       # discount factor for the cost function
    N: np.ndarray           # prediction horizon (number of steps)


def get_default_car_params() -> CarParams:
    """Parameter values in the original example."""
    return CarParams(
        m=np.array([0.043]),
        l=np.array([1/15.5]),
        lr=np.array([1/15.5/2]),

        cm1=np.array([0.28]),
        cm2=np.array([0.05]),
        cr0=np.array([0.011]),  # switched with cr2 in the paper, values taken from the code
        cr2=np.array([0.006]),  # see also Table I here: https://cdn.syscop.de/publications/Verschueren2016b.pdf
        cr3=np.array([5.0]),

        q_diag_sqrt=np.sqrt([
            1e-1,
            1e-8,
            1e-8,
            1e-8,
            1e-3,
            5e-3,
        ]),

        r_diag_sqrt=np.sqrt([
            1e-3,
            5e-3,
        ]),

        qe_diag_sqrt=np.sqrt([
            5e+0,
            1e+2,
            1e-8,
            1e-8,
            1e-3,
            5e-3,
        ]),

        slack_n_linear=np.array([100.0]),
        slack_n_quadratic=np.array([1.0]),
        slack_acc_linear=np.array([100.0]),
        slack_acc_quadratic=np.array([1.0]),

        a_lat_max=np.array([+4.0]),
        a_lat_min=np.array([-4.0]),
        a_long_max=np.array([+4.0]),
        a_long_min=np.array([-4.0]),
        n_max=np.array([+0.12]),
        n_min=np.array([-0.12]),

        D_max=np.array([+1.0]),
        D_min=np.array([-1.0]),
        delta_max=np.array([+0.4]),
        delta_min=np.array([-0.4]),
        dD_max=np.array([+10.0]),
        dD_min=np.array([-10.0]),
        ddelta_max=np.array([+2.0]),
        ddelta_min=np.array([-2.0]),

        dt=np.array([0.02]),
        gamma=np.array([1.0]),
        N=np.array([50]),
    )


def create_acados_params(car_params: CarParams, dimensionless: bool) -> tuple[AcadosParameter, ...]:
    """Create a list of parameters to be handled within acados"""
    from race_car.utils.scaling import get_cost_matrices, get_transformation_matrices
    # add the learnable cost weights and non-learnable progress reference
    # TODO: check cost bounds for the dimensionless case
    cost_min = 1e-12  # lower limit on the individual weights
    cost_max = 1e+6  # upper limit on the individual weights

    # nondimensionalize the cost weights if needed
    if dimensionless:
        Q, R, Qe = get_cost_matrices(car_params)
        Mx, Mu, _ = get_transformation_matrices(car_params)
        M = Mx.T @ Q @ Mx
        q_diag_sqrt_dimensionless = np.sqrt(M.diagonal())
        M = Mu.T @ R @ Mu
        r_diag_sqrt_dimensionless = np.sqrt(M.diagonal())
        M = Mx.T @ Qe @ Mx
        qe_diag_sqrt_dimensionless = np.sqrt(M.diagonal())

    params = [
        AcadosParameter(
            name="q_diag_sqrt",
            default=car_params.q_diag_sqrt if not dimensionless else q_diag_sqrt_dimensionless,
            space=gym.spaces.Box(
                low=np.sqrt(cost_min * np.ones_like(car_params.q_diag_sqrt)),
                high=np.sqrt(cost_max * np.ones_like(car_params.q_diag_sqrt)),
                dtype=np.float64),
            interface="learnable",
        ),
        AcadosParameter(
            name="r_diag_sqrt",
            default=car_params.r_diag_sqrt if not dimensionless else r_diag_sqrt_dimensionless,
            space=gym.spaces.Box(
                low=np.sqrt(cost_min * np.ones_like(car_params.r_diag_sqrt)),
                high=np.sqrt(cost_max * np.ones_like(car_params.r_diag_sqrt)),
                dtype=np.float64),
            interface="learnable",
        ),
        AcadosParameter(
            name="qe_diag_sqrt",
            default=car_params.qe_diag_sqrt if not dimensionless else qe_diag_sqrt_dimensionless,
            space=gym.spaces.Box(
                low=np.sqrt(cost_min * np.ones_like(car_params.qe_diag_sqrt)),
                high=np.sqrt(cost_max * np.ones_like(car_params.qe_diag_sqrt)),
                dtype=np.float64),
            interface="learnable",
        ),
        AcadosParameter(
            "sref",
            default=np.array([0.0]),  # scalar here, will be changed along the horizon at runtime
            interface="non-learnable",
            end_stages=list(range(car_params.N.item() + 1)),
        )
    ]
    return tuple(params)


def get_default_car_params_tracking() -> CarParams:
    """Similar to the original example, but with more emphasis on centerline tracking."""
    car_params = get_default_car_params()
    car_params.q_diag_sqrt = np.sqrt([5, 1e3, 1e3, 1, 1e-3, 5e-3])
    car_params.qe_diag_sqrt = car_params.q_diag_sqrt
    return car_params


def get_large_car_params() -> CarParams:
    """Returns the parameters of a specific larger car."""
    from race_car.utils.scaling import get_similar_car_params
    return get_similar_car_params(
        reference_params=get_default_car_params(),
        new_length=4.0,
        new_mass=1500.0,
        new_cr3=0.4,  # chosen approximately, top speed of small car is 2.5 m/s
    )


def get_large_car_params_tracking() -> CarParams:
    """Similar to the large car params, but with more emphasis on centerline tracking."""
    from race_car.utils.scaling import get_similar_car_params
    default_car_params = get_default_car_params_tracking()
    large_car_params = get_similar_car_params(
        reference_params=default_car_params,
        new_length=4.0,
        new_mass=1500.0,
        new_cr3=0.4,  # chosen approximately, top speed of small car is 2.5 m/s
    )
    return large_car_params


if __name__ == "__main__":
    from pprint import pprint
    car_params = get_default_car_params()
    pprint(car_params)
    get_large_car_params()
    get_large_car_params_tracking()
    print("-"*50)
    pprint(create_acados_params(car_params=car_params, dimensionless=True))
    print("ok")