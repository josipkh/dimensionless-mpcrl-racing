"""Estimates the closed-loop cost sensitivity w.r.t. plant-model mismatch, following Section II.A in https://ieeexplore.ieee.org/abstract/document/10938338"""
from copy import deepcopy
import numpy as np
import casadi as ca

from acados_template.utils import is_empty

from sensitivity_analysis.closed_loop_performance.sensitivity_utils import (
    create_solvers, solve_ocp_and_compute_sens, create_one_step_sensitivity_function
)
from race_car.acados_ocp import export_ocp
from race_car.utils.config import CarParams
from race_car.utils.track import get_track


def simulate_closed_loop(car_params: CarParams) -> tuple[dict, float]:
    """Runs a dimensionless closed-loop simulation, returns the state log and the total cost."""
    ocp = export_ocp(car_params=car_params, dimensionless=True)
    
    # set the reference progress offset
    # will be added to the current state online
    N = car_params.N.item()
    s_offset = np.linspace(
        start=0.0,
        stop=car_params.s_ref_N.item() / car_params.l.item(),
        num=N + 1,
    )

    nx = ocp.model.x.size1()
    nu = ocp.model.u.size1()
    track_length = get_track(car_params)[0][-1] / car_params.l.item()  # dimensionless track length

    # prepare the solvers and the integrator
    print("Setting up the solvers...")
    ocp_solver, sensitivity_solver = create_solvers(ocp)
    simulator = create_one_step_sensitivity_function(model=ocp.model, dt=ocp.solver_options.tf / ocp.solver_options.N_horizon)

    # define the closed-loop cost function
    ny = nx + nu
    yref = ca.SX.sym('yref', ny)
    temp_ocp = deepcopy(ocp)
    temp_ocp.translate_cost_to_external_cost(yref=yref)
    cost_fcn = ca.Function('cost_fcn', [temp_ocp.model.x, temp_ocp.model.u, yref], [temp_ocp.model.cost_expr_ext_cost], ['x_k', 'u_k', 'yref_k'], ['L'])
    yref_num = np.zeros(ny)
    yref_num[0] = track_length
    closed_loop_cost = 0.0

    # prepare the logs
    N_sim = 500
    log = {}
    log["x"] = np.zeros((N_sim + 1, nx))
    log["u"] = np.zeros((N_sim, nu))
    log["x"][0, :] = np.zeros(nx)

    log["du_dx"] = np.zeros((N_sim, nu, nx))
    log["df_dx"] = np.zeros((N_sim, nx, nx))
    log["df_du"] = np.zeros((N_sim, nx, nu))
    log["df_dp"] = np.zeros((N_sim, nx, ocp.model.p_global.size1()))


    print("Running the closed-loop simulation...")
    for k in range(N_sim):
        x_current = log["x"][k, :]

        # update reference (progress along centerline)
        s_ref = x_current[0] + s_offset
        for j in range(1, ocp.solver_options.N_horizon + 1):
            yref = np.concatenate(( np.array([s_ref[j]]), np.zeros(7 if j != ocp.solver_options.N_horizon else 5) ))
            ocp_solver.cost_set(j, "yref", yref)
            sensitivity_solver.cost_set(j, "yref", yref)

        u0_opt, du0_opt_dx0 = solve_ocp_and_compute_sens(ocp_solver, sensitivity_solver, x_current)
        x_next, df_dx, df_du, df_dp = simulator(x_current, u0_opt, ocp.p_global_values)

        log["x"][k + 1, :] = np.array(x_next).flatten()
        log["u"][k, :] = u0_opt
        log["df_dx"][k, :, :] = np.array(df_dx)
        log["df_du"][k, :, :] = np.array(df_du)
        log["df_dp"][k, :, :] = np.array(df_dp)
        log["du_dx"][k, :, :] = np.array(du0_opt_dx0)

        closed_loop_cost += cost_fcn(x_current, u0_opt, yref_num).full().item()  # TODO: implement for cartpole

        # check for termination
        if x_next[0] > track_length:
            log["x"] = log["x"][:k+1, :]
            log["u"] = log["u"][:k, :]
            log["df_dx"] = log["df_dx"][:k, :, :]
            log["df_dp"] = log["df_dp"][:k, :, :]
            log["du_dx"] = log["du_dx"][:k, :, :]
            break

    return log, closed_loop_cost


def calculate_closed_loop_cost_sensitivity(car_params: CarParams, log: dict) -> np.ndarray:
    """Builds the closed-loop cost sensitivity w.r.t. the parameters in the simulator (plant-model mismatch)."""
    ocp = export_ocp(car_params=car_params, dimensionless=True)
    N_sim = log["u"].shape[0]
    nx = ocp.model.x.size1()
    np_global = ocp.model.p_global.size1()
    nu = ocp.model.u.size1()
    ny = nx + nu

    yref = ca.SX.sym('yref', ny)

    # Jacobians of the stage cost w.r.t. state and control
    if is_empty(ocp.model.cost_expr_ext_cost):
        ocp.translate_cost_to_external_cost(yref=yref)

    dL_dx = ca.Function(
        'dL_dx',
        [ocp.model.x, ocp.model.u, yref],
        [ca.jacobian(ocp.model.cost_expr_ext_cost, ocp.model.x)],
        ['x_k', 'u_k', 'yref_k'],
        ['dL_dx_k']
    )

    dL_du = ca.Function(
        'dL_du',
        [ocp.model.x, ocp.model.u, yref],
        [ca.jacobian(ocp.model.cost_expr_ext_cost, ocp.model.u)],
        ['x_k', 'u_k', 'yref_k'],
        ['dL_du_k']
    )

    # closed-loop cost function
    L = ca.Function('L', [ocp.model.x, ocp.model.u, yref], [ocp.model.cost_expr_ext_cost], ['x_k', 'u_k', 'y_ref'], ['L'])

    Pi_k = np.zeros((nx, np_global))  # dx_k/dp (state sensitivity)
    Gamma = np.zeros((1, np_global))  # dL/dp (closed-loop cost sensitivity)

    # set the numerical data for the reference
    yref_num = np.zeros(ny)
    yref_num[0] = get_track(car_params)[0][-1] / car_params.l.item()  # dimensionless track length
    L_cl = 0.0

    for k in range(N_sim):
        x_k = log["x"][k, :]
        u_k = log["u"][k, :]
        df_dx_k = log["df_dx"][k, :, :]
        df_du_k = log["df_du"][k, :, :]
        df_dp_k = log["df_dp"][k, :, :]
        du_dx_k = log["du_dx"][k, :, :]

        dL_dx_k = np.array(dL_dx(x_k, u_k, yref_num).full())
        dL_du_k = np.array(dL_du(x_k, u_k, yref_num).full())

        Theta_k = du_dx_k @ Pi_k  # eq. (9)
        Gamma += dL_dx_k @ Pi_k + dL_du_k @ Theta_k  # eq. (10)
        Pi_k = (df_dx_k + df_du_k @ du_dx_k) @ Pi_k + df_dp_k  # eq. (8)

        L_cl += L(x_k, u_k, yref_num).full()

    # convert to relative sensitivity (divide by nominal values)
    Gamma_scaled = Gamma * ocp.p_global_values / L_cl

    # normalize to unit sum, keep signs
    Gamma_norm = Gamma_scaled / np.sum(np.abs(Gamma_scaled))

    return Gamma_norm.flatten()


# TODO: remove test
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
    from race_car.utils.config import get_default_car_params, get_large_car_params

    car_params = get_default_car_params()

    log, closed_loop_cost = simulate_closed_loop(car_params)
    N_sim = log["u"].shape[0]

    # Mx, Mu, _ = get_transformation_matrices(car_params)
    # plot_results_track(state_log=log["x"] @ Mx, car_params=car_params, total_time=N_sim * car_params.dt.item())
    # plot_results_classic(simX=log["x"][:N_sim] @ Mx, simU=log["u"] @ Mu, t=np.linspace(0.0, N_sim * car_params.dt.item(), N_sim), car_params=car_params)

    print("Calculating the closed-loop sensitivity...")
    dJ_dp = calculate_closed_loop_cost_sensitivity(car_params, log)

    print("dJ_dp:")
    with np.printoptions(precision=3, suppress=True):
        print(dJ_dp*100)
