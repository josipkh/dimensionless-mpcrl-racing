"""Sensitivity analysis of dynamic similarity w.r.t. the reference race car Pi-groups, based on activity scores https://www.sciencedirect.com/science/article/pii/S0951832016303052."""
import numpy as np
import casadi as ca
from scipy.stats import qmc
from tqdm import tqdm

from acados_template import AcadosModel

from race_car.utils.config import SymbolicCarParams, get_default_car_params
from race_car.utils.scaling import get_pi_groups, get_transformation_matrices
from race_car.utils.track import get_track
from race_car.model import car_model_sim


def create_sensitivity_function(model: AcadosModel, dt: float, N_horizon: int = 1) -> ca.Function:
    """A CasADi function for calculating dxk/dp along the horizon"""
    p_sym = ca.vertcat(model.u, model.p_global)
    ode = {'x': model.x, 'p': p_sym, 'ode': model.f_expl_expr}

    one_step_simulator = ca.integrator('one_step_sim', 'cvodes', ode, 0.0, dt)

    x_init = ca.SX.sym('x_init', model.x.size1())
    U_horizon = ca.SX.sym('U_horizon', model.u.size1(), N_horizon)
    p_global = model.p_global

    x_current = x_init
    trajectory_list = []

    for k in range(N_horizon):
        u_k = U_horizon[:, k]
        p_step = ca.vertcat(u_k, p_global)
        
        res = one_step_simulator(x0=x_current, p=p_step)
        x_current = res['xf']
        
        trajectory_list.append(x_current)

    X_horizon = ca.hcat(trajectory_list)  # (nx, N_horizon)
    X_flat = ca.reshape(X_horizon, -1, 1)  # (nx * N_horizon, 1)
    Jac_flat = ca.jacobian(X_flat, p_global)  # (nx * N_horizon, n_p)
    horizon_sens_fcn = ca.Function('horizon_sens', 
                                [x_init, U_horizon, p_global], 
                                [Jac_flat, X_horizon])
    return horizon_sens_fcn


def calculate_activity_scores(
        model: AcadosModel,
        dt: float,
        x_max: np.ndarray,
        x_min: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
        pi_nominal: np.ndarray,
        N_horizon: int = 1,
        N: int = int(1e3),
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:

    # NOTE: we assume that model.p_global contains only Pi-groups

    p_min = p_max = []  # NOTE: no variation in pi-groups for now
    sens_fcn = create_sensitivity_function(model, dt)

    # generate samples from typical variable/parameter bounds
    u_bounds = np.hstack((x_max, np.array(u_max).repeat(N_horizon), p_max))
    l_bounds = np.hstack((x_min, np.array(u_min).repeat(N_horizon), p_min))

    sampler = qmc.LatinHypercube(d=l_bounds.shape[0], rng=seed)
    samples = sampler.random(n=N)
    samples = qmc.scale(samples, l_bounds, u_bounds)

    # construct a gradient covariance matrix (active subspace method)
    nx = model.x.size1()
    npi = model.p_global.size1()
    x_range = np.tile(x_max - x_min, N_horizon)  # (nx * N_horizon,)
    C = np.zeros((npi, npi))
    for i in tqdm(range(N), desc="Progress"):
        jac_flat, _ = sens_fcn(samples[i][:nx], samples[i][nx:], pi_nominal)
        S_k = np.array(jac_flat)  # (nx * N_horizon, npi)

        # to account for the pi-group magnitude (use pi_sample if changing)
        S_k *= pi_nominal
        # account also for the range of the states for an accurate (relative) estimate
        S_k /= x_range[:, None]

        C += S_k.T @ S_k
    C /= N
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # returns values in ascending order

    # reverse the arrays to get descending order
    # (below eq. 14 in https://www.sciencedirect.com/science/article/pii/S0951832016303052)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # eigenvalues/eigenvectors don't represent specific pi-groups, use activity scores
    # alternatively, look at eigenvalues/eigenvectors to draw conclusions about parameter combinations

    # assuming 'eigenvalues' has shape (npi,) and 'eigenvectors' has shape (npi, npi)
    # and eigenvectors[:, i] is the i-th eigenvector
    activity_scores = eigenvectors ** 2 @ eigenvalues
    relative_group_importance = activity_scores / np.sum(eigenvalues)  # normalize values

    for i in range(relative_group_importance.shape[0]):
        print(f"Relative importance of pi_{i+1} (activity scores): {100*relative_group_importance[i]:.2f} %")

    return activity_scores, relative_group_importance


if __name__ == "__main__":
    # model and settings
    model = car_model_sim(car_params=SymbolicCarParams(), dimensionless=True)
    
    default_car_params = get_default_car_params()
    pi_nominal = get_pi_groups(default_car_params)
    Mx, _, mt = get_transformation_matrices(default_car_params)
    dt = default_car_params.dt.item() / mt.item()

    # dimensionless state limits (s, n, alpha, v)
    x_max_dimensional = np.array([
        get_track(default_car_params)[0][-1],
        default_car_params.n_max.item(),
        np.pi,
        default_car_params.v_max.item(),
    ])
    x_min_dimensional = np.array([
        0.0,
        default_car_params.n_min.item(),
        -np.pi,
        0.0,
    ])
    Mx_inv = np.linalg.inv(Mx[:4, :4])
    x_max = Mx_inv @ x_max_dimensional
    x_min = Mx_inv @ x_min_dimensional
    
    # dimensionless input limits (D, delta; no scaling needed)
    u_max = np.array([default_car_params.D_max, default_car_params.delta_max])
    u_min = np.array([default_car_params.D_min, default_car_params.delta_min])

    calculate_activity_scores(model, dt, x_max, x_min, u_max, u_min, pi_nominal)
