"""Defines the parameters of the race car OCP and their default values."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import casadi as ca

from leap_c.ocp.acados.parameters import AcadosParameter


@dataclass(kw_only=True)
class CarParams:
    # Chassis parameters
    m: np.ndarray   # [kg] mass of the car
    l: np.ndarray   # [m] length of the car
    lr: np.ndarray  # [m] distance from CG to rear axle

    # Longitudinal force parameters (eq. 4)
    cm1: np.ndarray  # [kgm/s^2] 
    cm2: np.ndarray  # [kg/s]
    cr0: np.ndarray  # [kgm/s^2]
    cr2: np.ndarray  # [kg/m]
    cr3: np.ndarray  # [s/m]

    # Cost weight matrices, defined by the decimal logarithm of their diagonal entries
    q_diag_log: np.ndarray   # stage cost, state residuals
    r_diag_log: np.ndarray   # stage cost, input residuals
    qe_diag_log: np.ndarray  # terminal cost, state residuals

    # Slack cost weights
    slack_n_linear: np.ndarray       # linear cost on the position constraint [1/m]
    slack_n_quadratic: np.ndarray    # quadratic cost on the position constraint [1/m^2]
    slack_acc_linear: np.ndarray     # linear cost on the acceleration constraint [1/(m/s^2)]
    slack_acc_quadratic: np.ndarray  # quadratic cost on the acceleration constraint [1/(m/s^2)^2]

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

    v_max: np.ndarray       # maximum long. speed [m/s] (approximate)
    s_ref_N: np.ndarray     # terminal progress reference [m]


@dataclass(kw_only=True)
class SymbolicCarParams:
    """Symbolic model variables, for definining purely dimensionless dynamics."""
    m: ca.SX = ca.SX.sym("m")
    l: ca.SX = ca.SX.sym("l")
    lr: ca.SX = ca.SX.sym("lr")
    cm1: ca.SX = ca.SX.sym("cm1")
    cm2: ca.SX = ca.SX.sym("cm2")
    cr0: ca.SX = ca.SX.sym("cr0")
    cr2: ca.SX = ca.SX.sym("cr2")
    cr3: ca.SX = ca.SX.sym("cr3")


def get_default_car_params() -> CarParams:
    """Parameter values based on the original example (notes when modified)."""
    return CarParams(
        m=np.array([0.043]),
        l=np.array([1/15.5]),
        lr=np.array([1/15.5/2]),

        cm1=np.array([0.28]),
        cm2=np.array([0.05]),
        cr0=np.array([0.011]),  # switched with cr2 in the paper, values taken from the code
        cr2=np.array([0.006]),  # see also Table I here: https://cdn.syscop.de/publications/Verschueren2016b.pdf
        cr3=np.array([5.0]),

        q_diag_log=np.log10([
            1e-2,
            1e-4,
            1e-4,
            1e-4,
            1e-3,
            5e-3,
        ]),

        # original values
        # q_diag_log=np.log10([
        #     1e-1,
        #     1e-8,
        #     1e-8,
        #     1e-8,
        #     1e-3,
        #     5e-3,
        # ]),

        r_diag_log=np.log10([
            1e-3,
            5e-3,
        ]),

        qe_diag_log=np.log10([
            5e+0,
            1e+2,
            1e-4,
            1e-4,
            1e-3,
            5e-3,
        ]),

        # original values
        # qe_diag_log=np.log10([
        #     5e+0,
        #     1e+2,
        #     1e-8,
        #     1e-8,
        #     1e-3,
        #     5e-3,
        # ]),

        slack_n_linear=np.array([100.0]),
        slack_n_quadratic=np.array([1.0]),
        slack_acc_linear=np.array([0.0]),  # original value: 100.0
        slack_acc_quadratic=np.array([1000.0]),  # original value: 1.0

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

        v_max=np.array([2.5]),
        s_ref_N = np.array([3.0]),
    )


def create_acados_params(car_params: CarParams, dimensionless: bool) -> tuple[AcadosParameter, ...]:
    """Create a list of leap-c parameters (learnable cost weights and non-learnable progress reference)."""
    from race_car.utils.scaling import get_cost_matrices, get_transformation_matrices
    # TODO: remove fixed parameters
    cost_min_default = 1e-5  # lower limit on the individual weights (for the default car)
    cost_max_default = 1e+4  # upper limit on the individual weights (for the default car)
    nx = 4  # original (non-extended) state dimension

    qx_diag_min_default = cost_min_default * np.ones_like(car_params.q_diag_log[:nx])
    qx_diag_max_default = cost_max_default * np.ones_like(car_params.q_diag_log[:nx])

    # scale the weight bounds (w.r.t. car size) for correct sampling
    default_car_params = get_default_car_params()
    Mx_default = get_transformation_matrices(default_car_params)[0][:nx, :nx]
    Mx = get_transformation_matrices(car_params)[0][:nx, :nx]
    scaling_factor = (Mx_default.diagonal() ** 2) / (Mx.diagonal() ** 2)
    qx_diag_min = qx_diag_min_default * scaling_factor
    qx_diag_max = qx_diag_max_default * scaling_factor

    if dimensionless:
        # learn dimensionless cost weights
        Q, R, Qe = get_cost_matrices(car_params)
        Mx, Mu, _ = get_transformation_matrices(car_params)
        M = Mx.T @ Q @ Mx
        q_diag_log = np.log10(M.diagonal())
        M = Mu.T @ R @ Mu
        r_diag_log = np.log10(M.diagonal())
        M = Mx.T @ Qe @ Mx
        qe_diag_log = np.log10(M.diagonal())

        # adjust the bounds as well
        qx_diag_min_log = np.log10(Mx.diagonal()[:nx] ** 2 * qx_diag_min)
        qx_diag_max_log = np.log10(Mx.diagonal()[:nx] ** 2 * qx_diag_max)
    else:
        q_diag_log = car_params.q_diag_log
        r_diag_log = car_params.r_diag_log
        qe_diag_log = car_params.qe_diag_log

        qx_diag_min_log = np.log10(qx_diag_min)
        qx_diag_max_log = np.log10(qx_diag_max)

    params = [
        AcadosParameter(
            name="qx_diag_log",
            default=q_diag_log[:nx],
            space=gym.spaces.Box(
                low=qx_diag_min_log,
                high=qx_diag_max_log,
                dtype=np.float64),
            interface="learnable",
        ),
        AcadosParameter(
            name="qu_diag_log",  # for penalizing control inputs in the extended state
            default=q_diag_log[nx:],
            interface="fix",  # tuning only the state cost weights is enough
        ),
        AcadosParameter(
            name="r_diag_log",
            default=r_diag_log,
            interface="fix",  # tuning only the state cost weights is enough
        ),
        AcadosParameter(
            name="qex_diag_log",
            default=qe_diag_log[:nx],
            space=gym.spaces.Box(  # use the same bounds as for the stage cost weights
                low=qx_diag_min_log,
                high=qx_diag_max_log,
                dtype=np.float64),
            interface="learnable",
        ),
        AcadosParameter(
            name="qeu_diag_log",  # for penalizing control inputs in the extended state
            default=qe_diag_log[nx:],
            interface="fix",  # tuning only the state cost weights is enough
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
    car_params.q_diag_log = np.log10([5, 1e3, 1e3, 1, 1e-3, 5e-3])
    car_params.qe_diag_log = car_params.q_diag_log
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
    print("Default car parameters:")
    car_params = get_default_car_params()
    pprint(car_params)
    print("-"*50)

    # dimensionless formulations should have the same default values and limits for the parameters
    for (param_a, param_b) in zip(
        create_acados_params(car_params=get_default_car_params(), dimensionless=True),
        create_acados_params(car_params=get_large_car_params(), dimensionless=True)
    ):
        assert np.allclose(param_a.default, param_b.default), f"Default values for parameter '{param_a.name}' do not match: {param_a.default} vs {param_b.default}"
        if param_a.space:
            assert np.allclose(param_a.space.high, param_b.space.high)
            assert np.allclose(param_a.space.low, param_b.space.low)

    for (param_a, param_b) in zip(
        create_acados_params(car_params=get_default_car_params_tracking(), dimensionless=True),
        create_acados_params(car_params=get_large_car_params_tracking(), dimensionless=True)
    ):
        assert np.allclose(param_a.default, param_b.default), f"Default values for parameter '{param_a.name}' do not match: {param_a.default} vs {param_b.default}"
        if param_a.space:
            assert np.allclose(param_a.space.high, param_b.space.high)
            assert np.allclose(param_a.space.low, param_b.space.low)

    create_acados_params(car_params=get_default_car_params(), dimensionless=False),
    create_acados_params(car_params=get_large_car_params(), dimensionless=False)
    create_acados_params(car_params=get_default_car_params_tracking(), dimensionless=False),
    create_acados_params(car_params=get_large_car_params_tracking(), dimensionless=False)

    SymbolicCarParams()

    print("ok")