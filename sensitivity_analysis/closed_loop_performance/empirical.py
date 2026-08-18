"""Tests the effect of plant-model mismatch on closed-loop performance using parameter perturbations."""

import os
import numpy as np
import casadi as ca
from acados_template import AcadosOcpSolver
from acados_template.utils import is_empty
from race_car.utils.log_utils import use_acados_temp_dir
from race_car.acados_ocp import export_ocp
from race_car.utils.config import CarParams
from race_car.utils.track import get_track
from race_car.utils.scaling import get_pi_groups


def evaluate_closed_loop(car_params: CarParams, pi_groups: np.ndarray) -> tuple[np.ndarray, float]:
    """Runs a closed-loop experiment, returns the state log and the total cost."""

    # recreate the OCP from scratch, to avoid conflicts with the previous run
    N = car_params.N.item()
    ocp = export_ocp(car_params=car_params, dimensionless=True)
    s_offset = np.linspace(
        start=0.0,
        stop=car_params.s_ref_N.item() / car_params.l.item(),
        num=N + 1,
    )
    track_length = get_track(car_params)[0][-1] / car_params.l.item()

    print("Setting up the solver...")
    use_acados_temp_dir(ocp)
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    # add plant-model mismatch
    pi_sim_values = pi_groups  # perturbed values of dimensionless groups
    ode = {'x': ocp.model.x, 'p': ca.vertcat(ocp.model.u, ocp.model.p_global), 'ode': ocp.model.f_expl_expr}
    dt_hat = ocp.solver_options.tf / ocp.solver_options.N_horizon
    simulator = ca.integrator('simulator', 'cvodes', ode, 0.0, dt_hat)

    # closed-loop cost function
    if is_empty(ocp.model.cost_expr_ext_cost):
        ny = ocp.model.x.size1() + ocp.model.u.size1()
        yref = ca.SX.sym('yref', ny)
        ocp.translate_cost_to_external_cost(yref=yref)
        yref_num = np.zeros(ny)
        yref_num[0] = track_length
    L = ca.Function('L', [ocp.model.x, ocp.model.u, yref], [ocp.model.cost_expr_ext_cost], ['x_k', 'u_k', 'yref_k'], ['L'])

    print("Running the closed-loop simulation...")
    x_current = np.zeros(ocp.model.x.shape)
    L_total = 0.0
    x_log = np.zeros((500 + 1, ocp.model.x.rows()))
    x_log[0, :] = x_current.flatten()
    k = 0
    while x_current[0] < track_length:  # dimensionless track length

        # update reference (progress along centerline)
        s_ref = x_current[0] + s_offset
        for j in range(1, N + 1):
            yref = np.concatenate(( np.array([s_ref[j]]), np.zeros(7 if j != N else 5) ))
            ocp_solver.cost_set(j, "yref", yref)

        # solve the OCP
        u0_opt = ocp_solver.solve_for_x0(x_current)
        status = ocp_solver.status
        if status != 0:
            ocp_solver.print_statistics()
            print(f"Solver failed with status {status}")
            breakpoint()

        # add to the total closed-loop cost
        L_total += L(x_current, u0_opt, yref_num)

        # simulate one step
        x_current = simulator(x0=x_current, p=ca.vertcat(u0_opt, pi_sim_values))['xf'].full()
        k += 1
        x_log[k, :] = x_current.flatten()

    x_log = x_log[:k, :]  # remove extra zeros

    return x_log, L_total.full().item()


def evaluate_grid(
    car_params: CarParams,
    param_indices: list[int],
    param_values: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    p1_grid, p2_grid = np.meshgrid(param_values[0], param_values[1])
    cost_grid = np.zeros_like(p1_grid, dtype=float)

    k = 0
    for i in range(p1_grid.shape[0]):
        for j in range(p1_grid.shape[1]):
            pi_variation = get_pi_groups(car_params)
            pi_variation[param_indices[0]] *= 1.0 + p1_grid[i, j]
            pi_variation[param_indices[1]] *= 1.0 + p2_grid[i, j]

            _, cost = evaluate_closed_loop(car_params, pi_variation)
            cost_grid[i, j] = cost

            print(f"Evaluation {k+1}/{p1_grid.size} done.")
            k += 1

    return p1_grid, p2_grid, cost_grid


def plot_grid_results(
    p1_grid: np.ndarray,
    p2_grid: np.ndarray,
    costs: np.ndarray,
    param_indices: tuple[int, int],
    output_file: str | None = None,
):
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

    idx1, idx2 = param_indices[0] + 1, param_indices[1] + 1  # use 1-based indexing for display
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

    if "DISPLAY" in os.environ:
        plt.show(block=False)
    else:
        output_path = output_file or "plot_sens_closed_loop.png"
        plt.savefig(output_path, dpi=300)
        print(f"saved plot to {output_path}")
        plt.close()


if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from race_car.acados_ocp import export_ocp
    from race_car.utils.config import get_default_car_params
    from race_car.utils.scaling import get_pi_groups

    car_params = get_default_car_params()
    pi_nominal = get_pi_groups(car_params)

    ### finite differences gradient check
    npi = np.shape(pi_nominal)[0]
    grad_fd = np.zeros(npi)
    h = 1e-4  # best guess from a sweep over h

    # calculate the baseline cost
    baseline_cost = evaluate_closed_loop(car_params, pi_nominal)[1]

    # load data if available
    try:
        grad_fd = np.load("output/grad_fd.npy")
    except OSError:
        for i in range(npi):
            print(f"Perturbing parameter {i+1}/{npi}...")
            
            pi_plus = np.copy(pi_nominal)
            pi_plus[i] *= (1.0 + h)
            _, J_plus = evaluate_closed_loop(car_params, pi_plus)
            print(f"J_plus: {J_plus}")

            pi_minus = np.copy(pi_nominal)
            pi_minus[i] *= (1.0 - h)
            _, J_minus = evaluate_closed_loop(car_params, pi_minus)
            print(f"J_minus: {J_minus}")

            dp_i = 2.0 * h * pi_nominal[i]
            grad_fd[i] = (J_plus - J_minus) / dp_i

        os.makedirs("output", exist_ok=True)
        np.save("output/grad_fd.npy", grad_fd)

    # convert to relative sensitivity:
    grad_scaled = (grad_fd * pi_nominal) / baseline_cost

    # normalize to show percentage importance (sums to 100)
    grad_norm = (grad_scaled / np.sum(np.abs(grad_scaled))) * 100

    with np.printoptions(suppress=True):
        print("Finite differences gradient (relative importance):")
        print(grad_norm)

