"""Test the effect of plant-model mismatch on closed-loop performance using parameter perturbations."""

import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.utils import is_empty
from race_car.utils.log_utils import use_acados_temp_dir
from race_car.acados_ocp import export_ocp
from race_car.utils.config import CarParams


def evaluate_closed_loop(car_params: CarParams, pi_variation: np.ndarray) -> float:
    """Runs a closed-loop experiment and returns the total cost."""

    N = car_params.N.item()
    ocp = export_ocp(car_params=car_params, dimensionless=True)
    s_offset = np.linspace(
        start=0.0,
        stop=car_params.s_ref_N.item() / car_params.l.item(),
        num=N + 1,
    )

    print("Setting up the solver...")
    use_acados_temp_dir(ocp)
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    # add plant-model mismatch
    p_sim_values = ocp.p_global_values * (1 + pi_variation)
    ode = {'x': ocp.model.x, 'p': ca.vertcat(ocp.model.u, ocp.model.p_global), 'ode': ocp.model.f_expl_expr}
    dt_hat = ocp.solver_options.tf / ocp.solver_options.N_horizon
    simulator = ca.integrator('simulator', 'cvodes', ode, 0.0, dt_hat)

    # closed-loop cost function
    if is_empty(ocp.model.cost_expr_ext_cost):
        ocp.translate_cost_to_external_cost()
    L = ca.Function('L', [ocp.model.x, ocp.model.u], [ocp.model.cost_expr_ext_cost], ['x_k', 'u_k'], ['L'])

    print("Running the closed-loop simulation...")
    x_current = np.zeros(ocp.model.x.shape)
    L_total = 0.0
    x_log = np.zeros((500 + 1, ocp.model.x.rows()))
    x_log[0, :] = x_current.flatten()
    k = 0
    while x_current[0] < 135.2:  # dimensionless track length

        # update reference (progress along centerline)
        s_ref = x_current[0] + s_offset
        for j in range(1, ocp.solver_options.N_horizon + 1):
            yref = np.concatenate(( np.array([s_ref[j]]), np.zeros(7 if j != ocp.solver_options.N_horizon else 5) ))
            ocp_solver.cost_set(j, "yref", yref)

        # solve the OCP
        u0_opt = ocp_solver.solve_for_x0(x_current)
        status = ocp_solver.status
        if status != 0:
            ocp_solver.print_statistics()
            print(f"Solver failed with status {status}")
            breakpoint()

        # add to the total closed-loop cost
        L_total += L(x_current, u0_opt)

        # simulate one step
        x_current = simulator(x0=x_current, p=ca.vertcat(u0_opt, p_sim_values))['xf'].full()
        k += 1
        x_log[k, :] = x_current.flatten()

    x_log = x_log[:k, :]  # remove extra zeros

    return x_log, L_total.full().item()


def evaluate_grid(
    car_params: CarParams,
    param_indices: list[int],
    param_ranges: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    p1_grid, p2_grid = np.meshgrid(param_ranges[0], param_ranges[1])
    cost_grid = np.zeros_like(p1_grid, dtype=float)

    k = 0
    for i in range(p1_grid.shape[0]):
        for j in range(p1_grid.shape[1]):
            pi_variation = np.zeros(5)
            pi_variation[param_indices[0]] = p1_grid[i, j]
            pi_variation[param_indices[1]] = p2_grid[i, j]

            # ocp.model.name = "model_" + str(k)
            _, cost = evaluate_closed_loop(car_params, pi_variation)
            cost_grid[i, j] = cost

            print(f"evaluation {k} done")
            k += 1

    return p1_grid, p2_grid, cost_grid


def plot_grid_results(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    costs: np.ndarray,
    param_indices: tuple[int, int] = (1, 2),
):
    idx1, idx2 = param_indices
    p1_flat = p1_grid.flatten()
    p2_flat = p2_grid.flatten()
    costs_flat = costs.flatten()

    # Find cost at nominal point (0, 0)
    nominal_mask = (p1_flat == 0) & (p2_flat == 0)
    if np.any(nominal_mask):
        cost_nominal = costs_flat[nominal_mask][0]
    else:
        # Fallback to point closest to (0,0) if exact origin isn't in grid
        origin_dist = p1_flat**2 + p2_flat**2
        cost_nominal = costs_flat[np.argmin(origin_dist)]

    # Compute relative percentage change: (J - J_nominal) / J_nominal * 100
    rel_cost_change = ((costs_flat - cost_nominal) / cost_nominal) * 100.0

    plt.figure(figsize=(11, 8.5))

    contour_filled = plt.tricontourf(
        p1_flat, p2_flat, rel_cost_change, levels=15, cmap="viridis", alpha=0.85
    )
    plt.tricontour(
        p1_flat, p2_flat, rel_cost_change, levels=15, colors="white", linewidths=0.5, alpha=0.6
    )

    plt.scatter(
        p1_flat, p2_flat, c=rel_cost_change, cmap="viridis", s=40, edgecolors="black", linewidths=0.5, zorder=3
    )

    plt.plot(0, 0, "rX", markersize=14, label=r"Nominal $\pi$", zorder=4)

    idx1, idx2 = param_indices
    xlabel = fr"Relative Perturbation in $\pi_{{{idx1}}}$ ($\Delta\pi_{{{idx1}}} / \pi_{{{idx1}}}$)"
    ylabel = fr"Relative Perturbation in $\pi_{{{idx2}}}$ ($\Delta\pi_{{{idx2}}} / \pi_{{{idx2}}}$)"
    title = fr"Relative Closed-Loop Cost Change vs. $\pi_{{{idx1}}}$ & $\pi_{{{idx2}}}$"

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)

    cbar = plt.colorbar(contour_filled)
    cbar.set_label(r"Relative Cost Change $\frac{\Delta J}{J_0} [\%]$", fontsize=12)

    plt.grid(True, linestyle=":", alpha=0.3, color="white")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params
    from race_car.acados_ocp import export_ocp
    import matplotlib.pyplot as plt
    
    car_params = get_default_car_params()

    # x_log, L = evaluate_closed_loop(car_params=car_params, pi_variation=np.zeros(5))
    # print(L)

    # from race_car.utils.plotting import plot_results_track
    # from race_car.utils.scaling import get_transformation_matrices
    # Mx = get_transformation_matrices(car_params)[0]
    # plot_results_track(state_log=x_log@Mx, car_params=car_params, total_time=0.0)

    p1_vals = np.linspace(-0.5, 0.5, 3)
    p2_vals = np.linspace(-0.5, 0.5, 3)
    param_indices = [1, 4]

    p1_grid, p2_grid, costs = evaluate_grid(
        car_params=car_params,
        param_indices=param_indices,
        param_ranges=(p1_vals, p2_vals),
    )

    plot_grid_results(p1_grid, p2_grid, costs, param_indices)
