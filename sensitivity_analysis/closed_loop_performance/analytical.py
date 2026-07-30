"""Estimates the closed-loop cost sensitivity w.r.t. plant-model mismatch, following Section II.A in https://ieeexplore.ieee.org/abstract/document/10938338"""
import numpy as np
import casadi as ca

from acados_template import AcadosOcp
from acados_template.utils import is_empty

from sensitivity_analysis.closed_loop_performance.sensitivity_utils import (
    create_solvers, solve_ocp_and_compute_sens, create_one_step_sensitivity_function
)
from race_car.acados_ocp import export_ocp
from race_car.utils.config import get_default_car_params, get_large_car_params


def simulate_closed_loop(ocp: AcadosOcp, x0: np.ndarray, N_sim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # if this is the race car example, extract progress offset and replace cost scaling
    race_car_example = False
    if ocp.solver_options.cost_scaling is not None:
        race_car_example = True
        s_offset = ocp.solver_options.cost_scaling
        ocp.solver_options.cost_scaling = np.ones(ocp.solver_options.N_horizon + 1)

    print("Setting up the solvers...")
    ocp_solver, sensitivity_solver = create_solvers(ocp)
    simulator = create_one_step_sensitivity_function(model=ocp.model, dt=ocp.solver_options.tf / ocp.solver_options.N_horizon)

    nx = ocp.model.x.size1()
    nu = ocp.model.u.size1()

    x_log = np.zeros((N_sim + 1, nx))
    u_log = np.zeros((N_sim, nu))
    x_log[0, :] = x0

    du_dx_log = np.zeros((N_sim, nu, nx))
    df_dx_log = np.zeros((N_sim, nx, nx))
    df_du_log = np.zeros((N_sim, nx, nu))
    df_dp_log = np.zeros((N_sim, nx, ocp.model.p_global.size1()))

    print("Running the closed-loop simulation...")
    for k in range(N_sim):
        x_current = x_log[k, :]

        if race_car_example:
            # update reference (progress along centerline)
            s_ref = x_current[0] + s_offset
            for j in range(1, ocp.solver_options.N_horizon + 1):
                yref = np.concatenate(( np.array([s_ref[j]]), np.zeros(7 if j != ocp.solver_options.N_horizon else 5) ))
                ocp_solver.cost_set(j, "yref", yref)
                sensitivity_solver.cost_set(j, "yref", yref)

        u0_opt, du0_opt_dx0 = solve_ocp_and_compute_sens(ocp_solver, sensitivity_solver, x_current)
        x_next, df_dx, df_du, df_dp = simulator(x_current, u0_opt, ocp.p_global_values)

        x_log[k + 1, :] = np.array(x_next).flatten()
        u_log[k, :] = u0_opt
        df_dx_log[k, :, :] = np.array(df_dx)
        df_du_log[k, :, :] = np.array(df_du)
        df_dp_log[k, :, :] = np.array(df_dp)
        du_dx_log[k, :, :] = np.array(du0_opt_dx0)

        if race_car_example:
            # check for termination
            if x_next[0] > 135.2:  # dimensionless track length
                x_log = x_log[:k, :]
                u_log = u_log[:k, :]
                df_dx_log = df_dx_log[:k, :, :]
                df_du_log = df_du_log[:k, :, :]
                df_dp_log = df_dp_log[:k, :, :]
                du_dx_log = du_dx_log[:k, :, :]
                break

    return x_log, u_log, df_dx_log, df_du_log, df_dp_log, du_dx_log


def calculate_closed_loop_cost_sensitivity(
        ocp: AcadosOcp,
        x_log: np.ndarray,
        u_log: np.ndarray,
        df_dx_log: np.ndarray,
        df_du_log: np.ndarray,
        df_dp_log: np.ndarray,
        du_dx_log: np.ndarray,
    ) -> np.ndarray:
    N_sim = u_log.shape[0]
    nx = ocp.dims.nx
    np_global = ocp.dims.np_global

    # Jacobians of the stage cost w.r.t. state and control
    if is_empty(ocp.model.cost_expr_ext_cost):
        ocp.translate_cost_to_external_cost()

    dL_dx = ca.Function(
        'dL_dx',
        [ocp.model.x, ocp.model.u],
        [ca.jacobian(ocp.model.cost_expr_ext_cost, ocp.model.x)],
        ['x_k', 'u_k'],
        ['dL_dx_k']
    )

    dL_du = ca.Function(
        'dL_du',
        [ocp.model.x, ocp.model.u],
        [ca.jacobian(ocp.model.cost_expr_ext_cost, ocp.model.u)],
        ['x_k', 'u_k'],
        ['dL_du_k']
    )

    # closed-loop cost function
    L = ca.Function('L', [ocp.model.x, ocp.model.u], [ocp.model.cost_expr_ext_cost], ['x_k', 'u_k'], ['L'])

    Pi_k = np.zeros((nx, np_global))  # dx_k/dp (state sensitivity)
    Gamma = np.zeros((1, np_global))  # dL/dp (closed-loop cost sensitivity)
    L_cl = 0  # total closed-loop cost

    for k in range(N_sim):
        x_k = x_log[k, :]
        u_k = u_log[k, :]
        df_dx_k = df_dx_log[k, :, :]
        df_du_k = df_du_log[k, :, :]
        df_dp_k = df_dp_log[k, :, :]
        du_dx_k = du_dx_log[k, :, :]

        dL_dx_k = np.array(dL_dx(x_k, u_k).full())
        dL_du_k = np.array(dL_du(x_k, u_k).full())

        Theta_k = du_dx_k @ Pi_k  # eq. (9)
        Gamma += dL_dx_k @ Pi_k + dL_du_k @ Theta_k  # eq. (10)
        Pi_k = (df_dx_k + df_du_k @ du_dx_k) @ Pi_k + df_dp_k  # eq. (8)

        L_cl += L(x_k, u_k)

    # convert to relative sensitivity (divide by nominal values)
    Gamma_scaled = Gamma * ocp.p_global_values / L_cl.full()

    # normalize to unit sum, keep signs
    Gamma_norm = Gamma_scaled / np.sum(np.abs(Gamma_scaled))

    return Gamma_norm.flatten()


def compare_solutions_and_sensitivities():
    ocp_large = export_ocp(car_params=get_large_car_params(), dimensionless=True)
    ocp_solver_large, sensitivity_solver_large = create_solvers(ocp_large)

    ocp_small = export_ocp(car_params=get_default_car_params(), dimensionless=True)
    ocp_solver_small, sensitivity_solver_small = create_solvers(ocp_small)

    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.5, 0.0])  # TODO: find a state with larger gradients
    u0_opt_large, du0_opt_dx0_large = solve_ocp_and_compute_sens(ocp_solver_large, sensitivity_solver_large, x0)
    u0_opt_small, du0_opt_dx0_small = solve_ocp_and_compute_sens(ocp_solver_small, sensitivity_solver_small, x0)

    print(u0_opt_large)
    print(du0_opt_dx0_large)
    print(u0_opt_small)
    print(du0_opt_dx0_small)

    assert np.allclose(u0_opt_large, u0_opt_small), "Optimal control inputs do not match"
    assert np.allclose(du0_opt_dx0_large, du0_opt_dx0_small), "Sensitivities do not match"

    print("Solutions and sensitivities match.")


if __name__ == "__main__":
    from race_car.utils.plotting import plot_results_classic, plot_results_track
    from race_car.utils.scaling import get_transformation_matrices
    from race_car.utils.config import get_default_car_params_tracking

    car_params = get_default_car_params_tracking()
    N = car_params.N.item()

    ocp = export_ocp(car_params=car_params, dimensionless=True)

    # set the reference progress offset
    # will be added to the current state online
    s_offset = np.linspace(
        start=0.0,
        stop=car_params.s_ref_N.item() / car_params.l.item(),
        num=N + 1,
    )
    ocp.solver_options.cost_scaling = s_offset  # pass using this field for convenience

    N_sim = 500
    x_log, u_log, df_dx_log, df_du_log, df_dp_log, du_dx_log = simulate_closed_loop(
        ocp=ocp,
        x0=np.zeros(ocp.model.x.rows()),
        N_sim=N_sim,
    )
    N_sim = x_log.shape[0]

    Mx, Mu, _ = get_transformation_matrices(car_params)

    plot_results_track(state_log=x_log@Mx, car_params=car_params, total_time=0.0)
    plot_results_classic(simX=x_log@Mx, simU=u_log@Mu, t=np.linspace(0.0, N_sim*car_params.dt.item(), N_sim), car_params=car_params)

    print("calculating the closed-loop sensitivity...")
    dJ_dp = calculate_closed_loop_cost_sensitivity(
        ocp=ocp,
        x_log=x_log,
        u_log=u_log,
        df_dx_log=df_dx_log,
        df_du_log=df_du_log,
        df_dp_log=df_dp_log,
        du_dx_log=du_dx_log
    )

    print("dJ_dp")
    with np.printoptions(precision=3, suppress=True):
        print(dJ_dp*100)
