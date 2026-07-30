"""Helper functions for creating differentiable MPC solvers and one-step integrators."""
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosCasadiOcpSolver, AcadosModel
from leap_c.ocp.acados.utils.create_solver import make_ocp_sensitivity_compatible
from race_car.utils.log_utils import use_acados_temp_dir


def create_solvers(ocp: AcadosOcp):
    """Uses a two-solver approach, based on https://arxiv.org/abs/2505.01353"""

    # create the forward solver
    use_acados_temp_dir(ocp)
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    # ocp_solver = AcadosCasadiOcpSolver(ocp, verbose=False)

    # create the sensitivity solver (du0_dp is not needed, only du0_dx0)
    make_ocp_sensitivity_compatible(ocp)
    ocp.code_gen_options.with_solution_sens_wrt_params = False
    ocp.code_gen_options.with_value_sens_wrt_params = False

    use_acados_temp_dir(ocp)

    ocp.model.name = 'sensitivity_solver'
    sensitivity_solver = AcadosOcpSolver(ocp, verbose=False)

    if ocp.dims.np_global > 0:
        ocp_solver.set_p_global_and_precompute_dependencies(ocp.p_global_values)
        sensitivity_solver.set_p_global_and_precompute_dependencies(ocp.p_global_values)

    return ocp_solver, sensitivity_solver


def solve_ocp_and_compute_sens(
        ocp_solver: AcadosOcpSolver | AcadosCasadiOcpSolver,
        sensitivity_solver: AcadosOcpSolver,
        x0: np.ndarray,
    ):
    """Calculate the sensitivity of the policy (u0_opt) w.r.t. initial state (given the optimal solution)."""

    u0_opt = ocp_solver.solve_for_x0(x0)
    status = ocp_solver.status
    if status != 0:
        ocp_solver.print_statistics()
        print(f"Solver failed with status {status}")
        breakpoint()

    iterate = ocp_solver.get_flat_iterate()

    sensitivity_solver.set_iterate(iterate)
    sensitivity_solver.setup_qp_matrices_and_factorize()

    if ocp_solver.status not in [0]:
        print(f"OCP solver returned status {ocp_solver.status}.")
        breakpoint()
    if sensitivity_solver.status not in [0, 2]:
        print(f"sensitivity solver returned status {sensitivity_solver.status}.")
        breakpoint()

    # Calculate the policy gradient
    du0_opt_dx0 = sensitivity_solver.eval_solution_sensitivity(
        stages=0,
        with_respect_to="initial_state",
        return_sens_x=False,
        return_sens_u=True,
        sanity_checks=True
    )["sens_u"]

    return u0_opt, du0_opt_dx0


def create_one_step_sensitivity_function(model: AcadosModel, dt: float) -> ca.Function:
    """
    Computes single-step sensitivities for the discrete-time transition x_{k+1} = f(x_k, u_k, p).
    
    Returns a CasADi function that computes:
      - x_next: Next state x_{k+1}
      - df_dx:  A = dx_{k+1} / dx_k  (nx, nx)
      - df_du:  B = dx_{k+1} / du_k  (nx, nu)
      - df_dp:  M = dx_{k+1} / dp    (nx, np)
    """
    ode = {'x': model.x, 'p': ca.vertcat(model.u, model.p_global), 'ode': model.f_expl_expr}
    one_step_simulator = ca.integrator('one_step_sim', 'cvodes', ode, 0.0, dt)

    x_k = ca.SX.sym('x_k', model.x.size1())
    u_k = ca.SX.sym('u_k', model.u.size1())
    p_global = model.p_global

    res = one_step_simulator(x0=x_k, p=ca.vertcat(u_k, p_global))
    x_next = res['xf']

    df_dx = ca.jacobian(x_next, x_k)       # Shape: (nx, nx)
    df_du = ca.jacobian(x_next, u_k)       # Shape: (nx, nu)
    df_dp = ca.jacobian(x_next, p_global)  # Shape: (nx, np)

    one_step_sens_fcn = ca.Function(
        'one_step_sens',
        [x_k, u_k, p_global],
        [x_next, df_dx, df_du, df_dp],
        ['x_k', 'u_k', 'p_global'],
        ['x_next', 'df_dx', 'df_du', 'df_dp']
    )

    return one_step_sens_fcn
