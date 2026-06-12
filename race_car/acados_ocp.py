"""Defines the acados OCP formulation for the race car problem."""
import time
import sys
import io
import contextlib

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpIterate, AcadosCasadiOcpSolver
import matplotlib.pyplot as plt

from race_car.model import car_model_ocp, export_acados_integrator
from race_car.utils.config import CarParams
from race_car.utils.plotting import plot_results_track, plot_results_classic, plot_lat_acc
from race_car.utils.scaling import get_cost_matrices, get_transformation_matrices
from race_car.utils.track import get_track
from race_car.utils.log_utils import create_acados_temp_dir

# settings for terminal constraints and slacks
from race_car.model import include_terminal_acc_constraint
use_slacks_on_acc_constraints = True
use_slacks_on_lat_position = True
include_terminal_position_constraint = True

# TODO: remove
from race_car.utils.ocp_scaling import ScaledAcadosOcpSolver
use_varying_reference = False  # adapt the progress reference based on the road curvature (tight turns vs. straights)


def export_ocp(car_params: CarParams, dimensionless: bool, soften_constraints: bool = True, promote_bounds: bool = False) -> AcadosOcp:
    """Exports the OCP for the given race car parameters."""
    ocp = AcadosOcp()

    # horizon and time step
    ocp.solver_options.N_horizon = int(car_params.N.item())
    ocp.solver_options.tf = car_params.dt.item() * ocp.solver_options.N_horizon

    # prediction model
    ocp.model = car_model_ocp(car_params=car_params, dimensionless=dimensionless)

    # constraints on states
    z = z_e = np.array([])  # linear cost weights for slacks
    Z = Z_e = np.array([])  # quadratic cost weights for slacks

    ocp.constraints.idxbx = np.array([1, 4, 5])
    ocp.constraints.lbx = np.array([car_params.n_min.item(), car_params.D_min.item(), car_params.delta_min.item()])
    ocp.constraints.ubx = np.array([car_params.n_max.item(), car_params.D_max.item(), car_params.delta_max.item()])

    ocp.constraints.idxbx_e = ocp.constraints.idxbx if include_terminal_position_constraint else ocp.constraints.idxbx[1:]
    ocp.constraints.lbx_e = ocp.constraints.lbx if include_terminal_position_constraint else ocp.constraints.lbx[1:]
    ocp.constraints.ubx_e = ocp.constraints.ubx if include_terminal_position_constraint else ocp.constraints.ubx[1:]

    if soften_constraints and use_slacks_on_lat_position:
        ocp.constraints.idxsbx = np.array([1])
        z = np.append(z, car_params.slack_n_linear.item())
        Z = np.append(Z, car_params.slack_n_quadratic.item())
        if include_terminal_position_constraint:
            ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
            z_e = np.append(z_e, car_params.slack_n_linear.item())
            Z_e = np.append(Z_e, car_params.slack_n_quadratic.item())

    # constraints on inputs
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([car_params.dD_min.item(), car_params.ddelta_min.item()])
    ocp.constraints.ubu = np.array([car_params.dD_max.item(), car_params.ddelta_max.item()])

    # constraints on nonlinear expressions (defined in the model)
    lh = np.array([car_params.a_long_min.item(), car_params.a_lat_min.item()])
    uh = np.array([car_params.a_long_max.item(), car_params.a_lat_max.item()])
    ocp.constraints.lh = lh.copy()
    ocp.constraints.uh = uh.copy()
    if include_terminal_acc_constraint:
        ocp.constraints.lh_e = lh.copy()
        ocp.constraints.uh_e = uh.copy()
    if soften_constraints and use_slacks_on_acc_constraints:
        nsh = lh.shape[0]
        ocp.constraints.idxsh = np.array(range(nsh))
        z = np.append(z, car_params.slack_acc_linear.item() * np.ones(nsh,))
        Z = np.append(Z, car_params.slack_acc_quadratic.item() * np.ones(nsh,))
        if include_terminal_acc_constraint:
            ocp.constraints.idxsh_e = ocp.constraints.idxsh
            z_e = np.append(z_e, car_params.slack_acc_linear.item() * np.ones(nsh,))
            Z_e = np.append(Z_e, car_params.slack_acc_quadratic.item() * np.ones(nsh,))

    if promote_bounds:
        # formulate (soft) box constraints on the state as (soft) general nonlinear constraints
        # bound slacks should go before the general constraint slacks (see links below)
        idxsbx = ocp.constraints.idxsbx
        idx_slacked = np.where(ocp.constraints.idxbx == idxsbx)
        ocp.model.con_h_expr = ca.vertcat(ocp.model.x[idxsbx], ocp.model.con_h_expr)
        ocp.model.con_h_expr_e = ca.vertcat(ocp.model.x[idxsbx], ocp.model.con_h_expr_e)
        ocp.constraints.lh = np.concatenate([ocp.constraints.lbx[idx_slacked], ocp.constraints.lh])
        ocp.constraints.uh = np.concatenate([ocp.constraints.ubx[idx_slacked], ocp.constraints.uh])
        ocp.constraints.lh_e = ocp.constraints.lh.copy()
        ocp.constraints.uh_e = ocp.constraints.uh.copy()
        ocp.constraints.idxsh = np.array(range(ocp.constraints.lh.shape[0]))
        ocp.constraints.idxsh_e = np.array(range(ocp.constraints.lh_e.shape[0]))

        # clear the fields related to the soft simple bounds
        ocp.constraints.idxbx = np.delete(ocp.constraints.idxbx, idx_slacked)
        ocp.constraints.lbx = np.delete(ocp.constraints.lbx, idx_slacked)
        ocp.constraints.ubx = np.delete(ocp.constraints.ubx, idx_slacked)
        ocp.constraints.idxbx_e = np.delete(ocp.constraints.idxbx_e, idx_slacked)
        ocp.constraints.lbx_e = np.delete(ocp.constraints.lbx_e, idx_slacked)
        ocp.constraints.ubx_e = np.delete(ocp.constraints.ubx_e, idx_slacked)
        ocp.constraints.idxsbx = np.array([])
        ocp.constraints.idxsbx_e = np.array([])

    # cost weights (NONLINEAR_LS), defined for the discrete-time cost
    # TODO: try using AUTO or LINEAR_LS
    Q, R, Qe = get_cost_matrices(car_params=car_params)
    ocp.solver_options.cost_scaling = np.ones(ocp.solver_options.N_horizon + 1)

    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_0 = ocp.model.u
    ocp.cost.yref_0 = np.zeros(ocp.model.cost_y_expr_0.shape)  # will not be changed later
    ocp.cost.W_0 = R
    
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.shape)  # will be changed at runtime
    ocp.cost.W = scipy.linalg.block_diag(Q, R)

    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.shape)  # will be changed at runtime
    ocp.cost.W_e = Qe

    # penalty on the slack variables (L2 for smoothness)
    # ordering: first state bounds, then general nonlinear constraints
    # see: https://discourse.acados.org/t/about-the-use-of-slack-variables/521
    # and: https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_solver.AcadosOcpSolver.get
    if soften_constraints and (use_slacks_on_acc_constraints or use_slacks_on_lat_position):
        ocp.cost.zl = ocp.cost.zu = z
        ocp.cost.Zl = ocp.cost.Zu = Z
        ocp.cost.zl_e = ocp.cost.zu_e = z_e
        ocp.cost.Zl_e = ocp.cost.Zu_e = Z_e

    ocp.constraints.x0 = np.zeros(ocp.model.x.shape)  # initial state, will be set in the MPC loop

    # solver options
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.tol = 1e-4
    # ocp.solver_options.nlp_solver_tol_stat = 1e-1  # stationarity residuals don't converge if this is lower

    # debugging TODO: remove
    ocp.solver_options.store_iterates = True
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"  # "FUNNEL_L1PEN_LINESEARCH"  # "MERIT_BACKTRACKING"  # add step correction
    # ocp.solver_options.globalization_full_step_dual = True  # when using funnel globalization
    # ocp.solver_options.print_level = 1  # to display iterate information
    # ocp.solver_options.qptol = ocp.solver_options.tol / 1000  # solve QPs more accurately
    # ocp.solver_options.nlp_solver_ext_qp_res = 1  # print out QP residuals too
    # ocp.solver_options.qpscaling_scale_objective = "OBJECTIVE_GERSHGORIN"
    # ocp.solver_options.qpscaling_scale_constraints = "INF_NORM"

    if dimensionless:
        ocp = nondimensionalize_ocp(ocp, car_params, soften_constraints, promote_bounds)

    return ocp


def nondimensionalize_ocp(ocp: AcadosOcp, car_params: CarParams, soften_constraints: bool, promote_bounds: bool) -> AcadosOcp:
    """Non-dimensionalizes the OCP formulation (cost and constraints)."""
    Mx, Mu, Mt = get_transformation_matrices(car_params=car_params)
    l = car_params.l.item()
    cr3 = car_params.cr3.item()

    # scale the time step
    dt_hat = car_params.dt.item() / Mt.item()
    ocp.solver_options.tf = dt_hat * ocp.solver_options.N_horizon

    # the prediction model is already dimensionless

    # scale the lateral position constraints
    if not promote_bounds:
        ocp.constraints.lbx[0] /= l
        ocp.constraints.ubx[0] /= l
        if include_terminal_position_constraint:
            ocp.constraints.lbx_e = ocp.constraints.lbx
            ocp.constraints.ubx_e = ocp.constraints.ubx

    # scale the input constraints
    ocp.constraints.lbu *= l * cr3
    ocp.constraints.ubu *= l * cr3

    # scale the nonlinear constraint expressions
    acc_scale = 1 / (l * cr3**2)
    if not promote_bounds:        
        ocp.constraints.lh /=  acc_scale
        ocp.constraints.uh /=  acc_scale
        if include_terminal_acc_constraint:
            ocp.constraints.lh_e /=  acc_scale
            ocp.constraints.uh_e /=  acc_scale
    else:
        h_scale = np.array([l, acc_scale, acc_scale])
        ocp.constraints.lh /=  h_scale
        ocp.constraints.uh /=  h_scale
        ocp.constraints.lh_e /=  h_scale
        ocp.constraints.uh_e /=  h_scale

    # scale the cost matrices
    ocp.cost.W_0 = Mu.T @ ocp.cost.W_0 @ Mu
    MxMu_block = scipy.linalg.block_diag(Mx, Mu)
    ocp.cost.W = MxMu_block.T @ ocp.cost.W @ MxMu_block
    ocp.cost.W_e = Mx.T @ ocp.cost.W_e @ Mx

    # scale the slack cost weights
    if soften_constraints:
        slack_scale = np.array([])
        slack_scale_e = np.array([])
        if use_slacks_on_lat_position:
            slack_scale = np.append(slack_scale, l)
            if include_terminal_position_constraint:
                slack_scale_e = np.append(slack_scale_e, l)
        if use_slacks_on_acc_constraints:
            slack_scale = np.append(slack_scale, acc_scale * np.ones((2,1)))
            if include_terminal_acc_constraint:
                slack_scale_e = np.append(slack_scale_e, acc_scale * np.ones((2,1)))

        ocp.cost.zl = ocp.cost.zu = ocp.cost.zl * slack_scale
        ocp.cost.Zl = ocp.cost.Zu = ocp.cost.Zl * slack_scale**2
        ocp.cost.zl_e = ocp.cost.zu_e = ocp.cost.zl_e * slack_scale_e
        ocp.cost.Zl_e = ocp.cost.Zu_e = ocp.cost.Zl_e * slack_scale_e**2

    # the initial state can remain unchanged, will be replaced anyway

    return ocp


def export_acados_ocp_solver(car_params: CarParams, dimensionless: bool) -> AcadosOcpSolver:
    """Creates an OCP solver based on the car parameters."""
    acados_ocp = export_ocp(car_params=car_params, dimensionless=dimensionless)

    # acados_ocp.code_gen_opts.code_export_directory = os.path.join("codegen", f"ocp_{car_params.l.item():.3g}".replace(".", "_"))
    # acados_ocp.code_gen_opts.code_export_directory += "_dimensionless" if dimensionless else ""
    # json_file = os.path.join(
    #         "json", 
    #         f"ocp_{car_params.l.item():.3g}".replace(".", "_")  # prevent overwriting
    #         + ("_dimensionless" if dimensionless else "")
    #         + ".json")

    # use a temporary directory to avoid overwriting files when running multiple experiments
    json_file = create_acados_temp_dir(acados_ocp)

    print("Setting up acados OCP solver...")
    acados_ocp_solver = AcadosOcpSolver(
        ocp=acados_ocp, 
        verbose=False,
        json_file=json_file,
    )
    return acados_ocp_solver


def export_acados_casadi_ocp_solver(car_params: CarParams, dimensionless: bool, solver: str) -> AcadosCasadiOcpSolver:
    """Creates an OCP solver based on the car parameters, which uses CasADi in the background."""
    acados_ocp = export_ocp(
        car_params=car_params,
        dimensionless=dimensionless,
        promote_bounds=True  # AcadosCasadiOcpSolver supports slacks only on general constraints at the moment
    )
    # casadi_solver_opts = {"print_time": False, "ipopt": {"print_level": 0}}
    casadi_solver_opts = {}

    print("Setting up acados CasADi OCP solver...")
    acados_casadi_ocp_solver = AcadosCasadiOcpSolver(
        ocp=acados_ocp, 
        verbose=False,
        solver=solver,
        casadi_solver_opts=casadi_solver_opts,
    )
    
    return acados_casadi_ocp_solver


def export_scaled_acados_ocp_solver(car_params: CarParams, dimensionless: bool) -> ScaledAcadosOcpSolver:
    ocp = export_ocp(car_params, dimensionless)
    s_max = get_track(car_params)[0][-1]  # total track length
    
    length_scale = car_params.l.item() if dimensionless else 1.0
    time_scale = get_transformation_matrices(car_params)[2].item() if dimensionless else 1.0

    # x = [s, n, alpha, v, D, delta]
    x_min = np.array([
        0.0,
        car_params.n_min.item() / length_scale,
        -np.pi,
        0.0,
        car_params.D_min.item(),
        car_params.delta_min.item(),
    ])

    x_max = np.array([
        s_max  / length_scale,
        car_params.n_max.item() / length_scale,
        np.pi,
        car_params.v_max.item() if not dimensionless else car_params.v_max.item() * (car_params.l.item() / car_params.dt.item()),
        car_params.D_max.item(),
        car_params.delta_max.item(),
    ])

    u_min = np.array([car_params.dD_min.item() * time_scale, car_params.ddelta_min.item() * time_scale])
    u_max = np.array([car_params.dD_max.item() * time_scale, car_params.ddelta_max.item() * time_scale])

    h_scale = ocp.constraints.uh  # already modified in the dimensionless case

    print("Setting up scaled acados OCP solver...")
    scaled_ocp_solver = ScaledAcadosOcpSolver(ocp, x_min, x_max, u_min, u_max, h_scale, verbose=False, json_file=create_acados_temp_dir(ocp))

    return scaled_ocp_solver


def get_varying_reference(s0: float, kappa_ref_s: ca.interpolant, ocp_solver: AcadosOcpSolver | AcadosCasadiOcpSolver | ScaledAcadosOcpSolver, car_params: CarParams) -> np.ndarray:
    """Adapts the progress reference based on the upcoming road curvature."""
    N = ocp_solver.ocp_solver.N if isinstance(ocp_solver, ScaledAcadosOcpSolver) else ocp_solver.N
    eps = 1e-4  # to avoid division by zero

    s_ref = np.zeros(N+1)
    s_ref[0] = s0
    for k in range(N):
        kappa_k = kappa_ref_s(s_ref[k])
        v_k = min(car_params.v_max.item(), np.sqrt(car_params.a_lat_max.item() / max(abs(kappa_k), eps)))
        d_k = v_k * car_params.dt.item()
        s_ref[k+1] = s_ref[k] + d_k

    return s_ref


def test_closed_loop(car_params: CarParams, dimensionless: bool, solver: str) -> None:
    """Runs a closed-loop simulation with the given car parameters."""
    if solver == "acados":
        ocp_solver = export_acados_ocp_solver(car_params=car_params, dimensionless=dimensionless)
    elif solver == "ipopt" or solver == "fatrop":
        ocp_solver = export_acados_casadi_ocp_solver(car_params=car_params, dimensionless=dimensionless, solver=solver)
    elif solver == "scaled":
        ocp_solver = export_scaled_acados_ocp_solver(car_params=car_params, dimensionless=dimensionless)
    else:
        raise ValueError("Unsupported solver.")

    # define the transformation functions
    Mx, Mu, _ = get_transformation_matrices(car_params=car_params)
    Mx_inv = np.linalg.inv(Mx)
    # Mu_inv = np.linalg.inv(Mu)
    dim2nondim_x = lambda x: Mx_inv @ x if dimensionless else x
    # nondim2dim_x = lambda x: Mx @ x if dimensionless else x
    # dim2nondim_u = lambda u: Mu_inv @ u if dimensionless else u
    nondim2dim_u = lambda u: Mu @ u if dimensionless else u

    # create the integrator for simulation (always dimensional)
    integrator = export_acados_integrator(car_params=car_params, dimensionless=False)

    # get the track data
    s_max = get_track(car_params)[0][-1]  # total track length
    sref_N = car_params.s_ref_N.item() / (car_params.l.item() if dimensionless else 1.0)  # terminal progress reference ("carrot")

    # preallocate memory for logging
    nx = ocp_solver.ocp.model.x.rows()
    nu = ocp_solver.ocp.model.u.rows()
    Nsim = 500  # from the original example
    simX = np.zeros((Nsim, nx))
    simU = np.zeros((Nsim, nu))
    s0 = 0.0  # start from standstill
    simX[0, 0] = s0
    tcomp_sum = 0
    tcomp_max = 0
    n_solver_fails = 0

    # form the curvature polynomial if needed
    if use_varying_reference:
        [sref, _, _, _, kapparef] = get_track(car_params=car_params)
        length = len(sref)
        sref = np.append(sref, [sref[length - 1] + sref[1:length]])
        kapparef = np.append(kapparef, kapparef[1:length])
        sref = np.append([-sref[length - 2] + sref[length - 81 : length - 2]], sref)
        kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)
        kappa_ref_s = ca.interpolant("kappa_ref_s", "bspline", [sref], kapparef)

    # closed-loop simulation
    N = ocp_solver.ocp.solver_options.N_horizon
    for i in range(Nsim-1):
        # set the initial condition
        ocp_solver.constraints_set(0, "lbx", dim2nondim_x(simX[i, :]))
        ocp_solver.constraints_set(0, "ubx", dim2nondim_x(simX[i, :]))

        # update reference (progress along centerline)
        s0 = dim2nondim_x(simX[i, :])[0]  # current track progress
        if use_varying_reference:
            s_ref = get_varying_reference(s0, kappa_ref_s, ocp_solver, car_params)
            if dimensionless:
                s_ref /= car_params.l.item()
        else:
            s_ref = s0 + sref_N * np.arange(0, N+1) / N  # equal spacing
        for j in range(1, N+1):  # for the intermediate stages
            ocp_solver.cost_set(j, "yref", np.concatenate(( [s_ref[j]], np.zeros(7 if j != N else 5) )))

        # solve the ocp
        t = time.time()

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            status = ocp_solver.solve()

        if (solver in ["acados", "fatrop", "scaled"] and status != 0) or (solver == "ipopt" and status != "Solve_Succeeded"):
            print("solver returned status {} in closed loop iteration {}.".format(status, i))
            n_solver_fails += 1

            print(out_buf.getvalue())
            print(err_buf.getvalue(), file=sys.stderr)

            # TODO: remove debugging code
            if False:
                # check the QP Hessian condition number
                hessian_conds = np.array([np.linalg.cond(ocp_solver.get_hessian_block(k)) for k in range(ocp_solver.N + 1)])
                max_hessian = np.max(hessian_conds, initial=0.0, where=np.isfinite(hessian_conds))
                max_hessian_stage = np.where(hessian_conds == max_hessian)[0][0]
                print(f"Maximum QP Hessian condition number: {max_hessian:.2e} at stage {max_hessian_stage}")

                # plot track with preview
                x = np.array(ocp_solver.get_iterate().x)
                plot_results_track(state_log=simX[:i,:],car_params=car_params,total_time=0.0, preview_log=x)

                # plot without preview
                plot_results_track(state_log=simX[:i,:],car_params=car_params,total_time=0.0)
                plot_results_classic(simX=simX[:i,:], simU=simU[:i,:], t=list(range(i)))

                # evaluate cost and constraints
                from acados_template.mpc_utils import AcadosCostConstraintEvaluator
                e = AcadosCostConstraintEvaluator(ocp_solver.ocp)
                e.update_all(ocp_solver)
                iterate = ocp_solver.get_iterate()
                X = iterate.x
                U = iterate.u
                x=X[0,:]
                u=U[0,:]
                eval_results = e.evaluate(x=x, u=u)
                eval_results["violation_soft_constraints"]
                eval_results["violation_hard_constraints"]

                # evaluate the acceleration constraints
                con_h_expr = ocp_solver.original_ocp.model.con_h_expr
                eval_acc = ca.Function('eval_acc', [ocp_solver.original_ocp.model.x, ocp_solver.original_ocp.model.u], [con_h_expr])
                res = eval_acc(x, u)
                print(f"Numerical Acc at stage 0: {res}")

                # check the jacobian -> all zeros (inputs are states)
                jac_h = ca.jacobian(con_h_expr, ocp_solver.original_ocp.model.u)
                eval_jac = ca.Function('h_jac', [ocp_solver.original_ocp.model.x, ocp_solver.original_ocp.model.u], [jac_h])
                print("Jacobian at stage 0:")
                print(eval_jac(x, u))
        else:
            print("solver succeeded in closed loop iteration {}.".format(i))
            if not isinstance(ocp_solver, ScaledAcadosOcpSolver):
                s = np.array(ocp_solver.get_iterate().x)[1:,0]
                sref = np.array([ocp_solver.cost_get(k, "yref")[0] for k in range(1,ocp_solver.N+1)])
                if any(sref < s):
                    print("reference went below the prediction")

        elapsed = time.time() - t

        # record timings
        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # get the optimal control input (part of the state in this formulation)
        u_opt = ocp_solver.get(1, "x")[-nu:]  # already dimensionless, no scaling needed

        # simulate one step with the integrator
        x_current = simX[i, :nx-nu]  # remove the extended states
        x_next = integrator.simulate(x=x_current, u=u_opt)

        # logging (inputs are control rates)
        simX[i+1, :-nu] = x_next
        simX[i+1, -nu:] = u_opt
        # simX[i+1, :] = ocp_solver.get(1, "x")  # take the ocp solution as the simulation
        simU[i, :] = nondim2dim_u(ocp_solver.get(0, "u"))  # save the physical values

        # check if one lap is done and break and remove entries beyond
        if x_next[0] > s_max + 0.1:
            # find where vehicle first crosses start line
            N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
            Nsim = i - N0  # correct to final number of simulation steps for plotting
            simX = simX[N0:i, :]
            simU = simU[N0:i, :]
            break
    
    # print some stats
    def format_number(x):
        if abs(x) >= 1:
            return "{:.2f}".format(x)
        else:
            return "{:.3g}".format(x)  # 3 significant digits
    
    print("Average computation time: {} s".format(format_number(tcomp_sum / Nsim)))
    print("Maximum computation time: {} s".format(format_number(tcomp_max)))
    print("Average speed: {} m/s".format(format_number(np.average(simX[:, 3]))))
    print("Lap time: {} s".format(format_number(Nsim * car_params.dt.item())))
    print("Number of solver fails: {} ({} %)".format(n_solver_fails, format_number(100*n_solver_fails/Nsim)))

    # plot the results
    t = np.linspace(0.0, Nsim * car_params.dt.item(), Nsim)
    plot_results_classic(simX, simU, t, car_params)
    plot_lat_acc(simX, t, car_params)
    plot_results_track(simX, car_params, t[-1])


def compare_iterates(it1: AcadosOcpIterate, it2: AcadosOcpIterate,
                     name1: str = "it1", name2: str = "it2"):
    it1 = it1.flatten()
    it2 = it2.flatten()
    print(f"comparing {name1} and {name2}:")
    diff_norm = (it1 - it2).inf_norm()
    print(f"  infinity norm of difference: {diff_norm:.6e}")

    print("  individual components:")
    for field in ['x', 'u', 'pi', 'lam']:
        diff_norm = np.linalg.norm(getattr(it1, field) - getattr(it2, field), np.inf)
        print(f"   {field}: {diff_norm:.6e}")
    print("")


def compare_formulation(car_params: CarParams, solver: str):
    """Compares the solution of a single OCP with the dimensional and dimensionless formulation."""
    if solver == "ipopt":
        ocp_solver_dimensional = export_acados_casadi_ocp_solver(car_params, dimensionless=False)
        ocp_solver_dimensionless = export_acados_casadi_ocp_solver(car_params, dimensionless=True)
    if solver == "acados":
        ocp_solver_dimensional = export_acados_ocp_solver(car_params, dimensionless=False)
        ocp_solver_dimensionless = export_acados_ocp_solver(car_params, dimensionless=True)

    N = car_params.N.item()
    l = car_params.l.item()
    s0 = 0.0
    sref_N = car_params.s_ref_N.item()
    for j in range(1, N):
        yref = np.array([s0 + sref_N * j / N, 0, 0, 0, 0, 0, 0, 0])
        ocp_solver_dimensional.set(j, "yref", yref)
        ocp_solver_dimensionless.set(j, "yref", yref / l)
    yref_N = np.array([s0 + sref_N, 0, 0, 0, 0, 0])
    ocp_solver_dimensional.set(N, "yref", yref_N)
    ocp_solver_dimensionless.set(N, "yref", yref_N / l)
    
    x0 = np.array([s0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp_solver_dimensional.solve_for_x0(x0)
    sol_dimensional = ocp_solver_dimensional.get_iterate()
    ocp_solver_dimensionless.solve_for_x0(x0)
    sol_dimensionless = ocp_solver_dimensionless.get_iterate()

    compare_iterates(it1=sol_dimensional, it2=sol_dimensionless, name1="dimensional", name2="dimensionless")

    Mx, Mu, _ = get_transformation_matrices(car_params)    
    x_dimensional = np.array(sol_dimensional.x)
    u_dimensional = np.array(sol_dimensional.u)
    x_dimensionless = np.array(sol_dimensionless.x) @ Mx
    u_dimensionless = np.array(sol_dimensionless.u) @ Mu

    print("x diff: ", np.max(np.abs(x_dimensional - x_dimensionless)))
    print("u diff: ", np.max(np.abs(u_dimensional - u_dimensionless)))

    n_iter_dimensional = ocp_solver_dimensional.get_stats('sqp_iter') if solver == "acados" else ocp_solver_dimensional.solver_stats["iter_count"]
    n_iter_dimensionless = ocp_solver_dimensionless.get_stats('sqp_iter') if solver == "acados" else ocp_solver_dimensionless.solver_stats["iter_count"]
    print("n_iter dimensional: ", n_iter_dimensional)
    print("n_iter dimensionless: ", n_iter_dimensionless)

    fig, axes = plt.subplots(8, 1, sharex=True, constrained_layout=True)
    x_labels = ocp_solver_dimensional.ocp.model.x_labels
    u_labels = ocp_solver_dimensional.ocp.model.u_labels
    labels = ["dimensional", "dimensionless"]
    states_list = [x_dimensional, x_dimensionless]
    controls_list = [u_dimensional, u_dimensionless]
    for k_axes in range(2):
        states = states_list[k_axes]
        controls = controls_list[k_axes]
        for i in range(6):
            axes[i].plot(states[:,i], linestyle='--' if k_axes == 1 else '-', color="b" if k_axes == 0 else "r", label=labels[k_axes])
            axes[i].grid(visible=True)
            if k_axes == 0:
                axes[i].set_ylabel(x_labels[i])
        for i in range(2):
            axes[6+i].stairs(controls[:,i], lw=1.5, linestyle='--' if k_axes == 1 else '-', color="b" if k_axes == 0 else "r")
            axes[6+i].grid(visible=True)
            if k_axes == 0:
                axes[6+i].set_ylabel(u_labels[i])
                # axes[6+i].axhline(ocp_solver_dimensional.ocp.constraints.lbu[i], color='k', linestyle='--', alpha=0.5)
                # axes[6+i].axhline(ocp_solver_dimensional.ocp.constraints.ubu[i], color='k', linestyle='--', alpha=0.5)
        axes[-1].set_xlabel('k')        
        axes[0].set_title("OCP solution comparison ({}, L = {:.2f} m)".format(solver, car_params.l.item()))
        axes[0].legend()
    fig.align_ylabels()
    plt.show(block=False)

    # plot side-by-side
    # fig, axes_array = plt.subplots(8, 2, sharex=True, constrained_layout=True, sharey='row')
    # x_labels = ocp_solver_dimensional.ocp.model.x_labels
    # u_labels = ocp_solver_dimensional.ocp.model.u_labels
    # states_list = [x_dimensional, x_dimensionless @ Mx]
    # controls_list = [u_dimensional, u_dimensionless @ Mu]
    # for k_axes in range(2):
    #     states = states_list[k_axes]
    #     controls = controls_list[k_axes]
    #     axes = axes_array[:,k_axes]

    #     for i in range(6):
    #         axes[i].plot(states[:,i])
    #         axes[i].grid(visible=True)
    #         if k_axes == 0:
    #             axes[i].set_ylabel(x_labels[i])
    #     for i in range(2):
    #         ctrl_tmp = controls[:,i]
    #         axes[6+i].stairs(np.append([ctrl_tmp[0]], ctrl_tmp), lw=1.5)
    #         axes[6+i].grid(visible=True)
    #         # if k_axes == 0:
    #         #     axes[6+i].set_ylabel(u_labels[i])
    #         # axes[-1].axhline(+F_max, color='r', linestyle='--', alpha=0.5)
    #         # axes[-1].axhline(-F_max, color='r', linestyle='--', alpha=0.5)
    #     axes[-1].set_xlabel('k')        
    #     axes[0].set_title("dimensionless OCP + scaling" if k_axes == 1 else "dimensional OCP")
    # fig.align_ylabels()
    # plt.show(block=False)


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params

    car_params = get_default_car_params()

    # compare_formulation(car_params, solver="ipopt")
    test_closed_loop(car_params=car_params, dimensionless=True, solver="acados")

    if plt.get_fignums():
        if hasattr(sys, 'ps1'):
            input("Press Enter to continue...")  # keep the plots open
        plt.close('all')