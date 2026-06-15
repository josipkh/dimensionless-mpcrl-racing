"""Defines the dynamics models for use in the OCP and the simulation."""

import os
from pathlib import Path
from tempfile import mkdtemp
from dataclasses import fields

import casadi as ca
import numpy as np
import sympy as sp
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from leap_c.ocp.acados.utils.delete_directory_hook import DeleteDirectoryHook

from race_car.utils.config import CarParams, SymbolicCarParams, get_default_car_params
from race_car.utils.scaling import get_transformation_matrices, get_pi_groups
from race_car.utils.track import get_track
from race_car.utils.symbolics import casadi2sympy, sympy2casadi
from race_car.utils.log_utils import create_acados_temp_dir

include_terminal_acc_constraint = True  # not used in the original example


def car_model_ocp(car_params: CarParams | SymbolicCarParams, dimensionless: bool) -> AcadosModel:
    """Get the prediction model for use in the OCP (control rate formulation)."""

    if dimensionless:
        # symbolic parameters needed for model reformulation
        numeric_car_params = car_params if isinstance(car_params, CarParams) else None  # save values for later substitution
        car_params = SymbolicCarParams()

    # load track parameters
    [sref, _, _, _, kapparef] = get_track(car_params=car_params)

    # race car parameters
    m = car_params.m[0]
    l = car_params.l[0]
    c1 = car_params.lr[0] / car_params.l[0]
    c2 = 1 / car_params.l[0]
    cm1 = car_params.cm1[0]
    cm2 = car_params.cm2[0]
    cr0 = car_params.cr0[0]
    cr2 = car_params.cr2[0]
    cr3 = car_params.cr3[0]

    # states
    s = ca.SX.sym("s")
    n = ca.SX.sym("n")
    alpha = ca.SX.sym("alpha")
    v = ca.SX.sym("v")
    D = ca.SX.sym("D")
    delta = ca.SX.sym("delta")
    x = ca.vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = ca.SX.sym("derD")
    derDelta = ca.SX.sym("derDelta")
    u = ca.vertcat(derD, derDelta)

    # state derivatives
    sdot = ca.SX.sym("sdot")
    ndot = ca.SX.sym("ndot")
    alphadot = ca.SX.sym("alphadot")
    vdot = ca.SX.sym("vdot")
    Ddot = ca.SX.sym("Ddot")
    deltadot = ca.SX.sym("deltadot")
    xdot = ca.vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    if dimensionless:
        # use a placeholder for curvature interpolant if the model is purely symbolic
        kapparef_s = ca.SX.sym("kapparef")
    else:
        # interpolate (dimensionless) curvature vs. progress
        # appropriate scaling is done in the physical model below
        sref /= l
        kapparef *= l
        # copy loop to the end (for preview close to the finish line)
        length = len(sref)
        sref = np.append(sref, [sref[length - 1] + sref[1:length]])
        kapparef = np.append(kapparef, kapparef[1:length])
        # copying to negative progess not needed when starting from zero
        # sref = np.append([-sref[length - 2] + sref[length - 81 : length - 2]], sref)
        # kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

        # compute spline interpolations (using normalized values for better conditioning)
        sref_min, sref_max = sref[0], sref[-1]  # assuming sref is sorted
        sref_normalized = (sref - sref_min) / (sref_max - sref_min)  # scale to range [0,1]
        kapparef_min, kapparef_max = np.min(kapparef), np.max(kapparef)
        kapparef_normalized = 2 * (kapparef - kapparef_min) / (kapparef_max - kapparef_min) - 1  # scale to range [-1,1]

        kapparef_s_normalized = ca.interpolant("kapparef_s", "bspline", [sref_normalized], kapparef_normalized)

        # redefine inputs/outputs to get proper physical values in the model
        s_dimensionless = s / l  # interpolant input; output is scaled in the expressions below
        s_normalized = (s_dimensionless - sref_min) / (sref_max - sref_min)
        kapparef_norm_evaluated = kapparef_s_normalized(s_normalized)
        kapparef_s = 0.5 * (kapparef_norm_evaluated + 1) * (kapparef_max - kapparef_min) + kapparef_min

    # dynamics
    # NOTE: kapparef_s in the expressions represents the dimensionless interpolant (required for subtitutions later)
    Fxd = (cm1 - cm2 * v) * D - cr2 * v**2 - cr0 * ca.tanh(cr3 * v)
    sdot_rhs = (v * ca.cos(alpha + c1 * delta)) / (1 - 1/l * kapparef_s * n)
    f_expl = ca.vertcat(
        sdot_rhs,
        v * ca.sin(alpha + c1 * delta),
        v * c2 * delta - 1/l * kapparef_s * sdot_rhs,
        Fxd / m * ca.cos(c1 * delta),
        derD,
        derDelta,
    )

    # constraints expressions (small sideslip angle is assumed)
    a_long = Fxd / m
    a_lat = Fxd * ca.sin(c1 * delta) / m + c2 * v**2 * delta
    con_h_expr = ca.vertcat(a_long, a_lat)

    # assign fields to AcadosModel
    model = AcadosModel()
    model.name = "car_model_ocp"
    model.x = x
    model.u = u
    model.xdot = xdot

    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.con_h_expr = con_h_expr
    if include_terminal_acc_constraint:
        model.con_h_expr_e = con_h_expr

    model.x_labels = [
        r"$s$ [m]",
        r"$n$ [m]",
        r"$\alpha$ [rad]",
        r"$v$ [m/s]",
        r"$D$ [-]",
        r"$\delta$ [rad]",
    ]
    model.u_labels = [
        r"$\dot{D}$ [1/s]",
        r"$\dot{\delta}$ [rad/s]",
    ]
    model.t_label = "$t$ [s]"

    # make dimensionless if needed, use given values for parameters
    if dimensionless:
        model = nondimensionalize_dynamics(car_params=car_params, model=model)
        model = replace_pi_groups(model=model, car_params=numeric_car_params)

    return model


def car_model_sim(car_params: CarParams, dimensionless: bool) -> AcadosModel:
    """Get the dynamics model for simulation, with the original states and inputs."""
    # get the OCP model
    model_ocp = car_model_ocp(car_params=car_params, dimensionless=dimensionless)

    # remove the control input rate from the model for simulation
    nx = 4
    nu = 2
    model = AcadosModel()
    model.f_impl_expr = model_ocp.f_impl_expr[0:nx]
    model.f_expl_expr = model_ocp.f_expl_expr[0:nx]
    model.x = model_ocp.x[0:nx]
    model.xdot = model_ocp.xdot[0:nx]
    model.u = model_ocp.x[nx:nx+nu]
    model.p_global = model_ocp.p_global  # keep the global parameters
    model.name = model_ocp.name.replace("ocp", "sim")
    model.x_labels = model_ocp.x_labels[0:nx]
    model.u_labels = model_ocp.x_labels[nx:nx+nu]
    model.t_label = model_ocp.t_label

    return model


def export_acados_integrator(car_params: CarParams, dimensionless: bool) -> AcadosSimSolver:
    """Create and return an acados integrator for simulating the car model."""

    acados_sim = AcadosSim()
    acados_sim.model = car_model_sim(car_params=car_params, dimensionless=dimensionless)
    acados_sim.solver_options.T = car_params.dt.item()
    if dimensionless:
        acados_sim.solver_options.T /= get_transformation_matrices(car_params)[2].item()  # scale the sampling time
    acados_sim.solver_options.integrator_type = "ERK"
    acados_sim.solver_options.num_stages = 4
    acados_sim.solver_options.num_steps = 3  # match the OCP integrator accuracy
    acados_sim.solver_options.sens_forw = False  # we don't need this information for simulations

    # assign a random directory to prevent overwriting
    json_file = create_acados_temp_dir(acados_sim)

    print("Setting up acados integrator...")
    acados_sim_solver = AcadosSimSolver(
        acados_sim=acados_sim, 
        verbose=False,
        json_file=json_file,
    )

    return acados_sim_solver


def nondimensionalize_dynamics(car_params: CarParams, model: AcadosModel) -> AcadosModel:
    """Convert the dynamics to the dimensionless form (without replacing the CasADi variables)."""
    x = model.x
    u = model.u
    f_expl = model.f_expl_expr
    con_h_expr = model.con_h_expr  # nonlinear constraints are handled similar to the dynamics

    Mx, Mu, mt = get_transformation_matrices(car_params=car_params)
    x_scale = np.diagonal(Mx)
    u_scale = np.diagonal(Mu)
    dx_scale = x_scale / mt[0]

    # RHS/constraints states
    for k in range(len(x_scale)):
        f_expl = ca.substitute(f_expl, x[k], x_scale[k] * x[k])
        con_h_expr = ca.substitute(con_h_expr, x[k], x_scale[k] * x[k])

    # RHS/constraints actions
    for k in range(len(u_scale)):
        f_expl = ca.substitute(f_expl, u[k], u_scale[k] * u[k])
        con_h_expr = ca.substitute(con_h_expr, u[k], u_scale[k] * u[k])

    # LHS (derivatives)
    for k in range(len(dx_scale)):
        f_expl[k] /= dx_scale[k]

    # scale the acceleration constraints to get dimensionless expressions
    # (eq. (19) in https://arxiv.org/abs/2512.08667)
    acc_scale = 1 / (car_params.l[0] * car_params.cr3[0]**2)
    con_h_expr /= acc_scale

    model.f_impl_expr = []  # TODO: implement dimensionless implicit dynamics
    model.f_expl_expr = f_expl
    model.con_h_expr = con_h_expr
    if include_terminal_acc_constraint:
        model.con_h_expr_e = con_h_expr
    model.name += "_dimensionless"

    model.x_labels = [
        r"$\hat{s}$ [p.u.]",
        r"$\hat{n}$ [p.u.]",
        r"$\hat{\alpha}$ [rad]",
        r"$\hat{v}$ [p.u.]",
        r"$\hat{D}$ [-]",
        r"$\hat{\delta}$ [rad]",
    ]
    model.u_labels = [
        r"$\dot{\hat{D}}$ [?]",
        r"$\dot{\hat{\delta}}$ [?]",
    ]
    model.t_label = r"$\hat{t}$ [-]"

    return model


def replace_pi_groups(model: AcadosModel, car_params: CarParams | None = None) -> None:
    """Introduces new CasADi variables for lumped expressions of physical parameters."""
    model.p_global = ca.vertcat(*[field.default for field in fields(SymbolicCarParams)])  # symbolic physical parameters
    kapparef_sym = ca.SX.sym("kapparef")  # symbol for the curvature interpolant
    pi_expressions_casadi = get_pi_groups(SymbolicCarParams())  # CasADi expressions for Pi-groups
    pi_groups_casadi = ca.SX.sym('pi', len(pi_expressions_casadi), 1)  # new CasADi variables, for substitution

    # create analogous SymPy symbols for all CasADi variables
    casadi_vars = ca.vertcat(model.x, model.u, model.p_global, kapparef_sym, pi_groups_casadi)
    sympy_dict = {casadi_vars[k].name(): sp.symbols(casadi_vars[k].name(), positive=True, real=True) for k in range(casadi_vars.numel())}

    # define substitutions avoiding fractions (helps internally with simplification)
    # NOTE: this part must be done manually (the rest is pretty generic)
    # pi_groups_sympy = get_pi_groups(sympy_dict)  # SymPy expressions for the Pi-groups
    substitutions = {
        sympy_dict["lr"]: sympy_dict["pi_0"] * sympy_dict["l"],
        sympy_dict["cm1"]: sympy_dict["pi_1"] * sympy_dict["m"] / (sympy_dict["cr3"]**2 * sympy_dict["l"]),
        sympy_dict["cm2"]: sympy_dict["pi_2"] * sympy_dict["m"] / (sympy_dict["cr3"] * sympy_dict["l"]),
        sympy_dict["cr0"]: sympy_dict["pi_3"] * sympy_dict["m"] / (sympy_dict["cr3"]**2 * sympy_dict["l"]),
        sympy_dict["cr2"]: sympy_dict["pi_4"] * sympy_dict["m"] / sympy_dict["l"]
    }

    # dynamics
    for k in range(model.x.shape[0]):        
        model.f_expl_expr[k] = substitute_expression_through_sympy(
            casadi_expr=model.f_expl_expr[k],
            casadi_vars=casadi_vars,
            sympy_dict=sympy_dict,
            substitutions=substitutions
        )

    # embed a dimensionless interpolant for the curvature
    [sref, _, _, _, kapparef] = get_track(car_params=get_default_car_params())
    l = get_default_car_params().l[0]
    sref /= l
    kapparef *= l

    # copy loop to the end (for preview close to the finish line)
    length = len(sref)
    sref = np.append(sref, [sref[length - 1] + sref[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])

    # compute spline interpolations (using normalized values for improved accuracy)
    sref_min, sref_max = sref[0], sref[-1]  # assuming sref is sorted
    sref_normalized = (sref - sref_min) / (sref_max - sref_min)  # scale to range [0,1]
    kapparef_min, kapparef_max = np.min(kapparef), np.max(kapparef)
    kapparef_normalized = 2 * (kapparef - kapparef_min) / (kapparef_max - kapparef_min) - 1  # scale to range [-1,1]
    kapparef_s_normalized = ca.interpolant("kapparef_s", "bspline", [sref_normalized], kapparef_normalized)

    # redefine inputs/outputs to get proper physical values in the model
    s_dimensionless = model.x[0]
    s_normalized = (s_dimensionless - sref_min) / (sref_max - sref_min)
    kapparef_norm_evaluated = kapparef_s_normalized(s_normalized)
    kapparef_s = 0.5 * (kapparef_norm_evaluated + 1) * (kapparef_max - kapparef_min) + kapparef_min

    # replace the symbolic variable with the interpolant (kapparef_sym -> kapparef_s)
    # have to go through a temp CasADi function, can't substitute an interpolant directly
    forward_map = ca.Function('forward_map', [model.x, model.u, pi_groups_casadi, kapparef_sym], [model.f_expl_expr])
    model.f_expl_expr = forward_map(model.x, model.u, pi_groups_casadi, kapparef_s)

    # try forming a CasADi function (it will automatically check for free variables)
    model.p_global = pi_groups_casadi  # only dimensionless variables should remain in the model
    ca.Function("dimensionless_dynamics", [model.x, model.u, model.p_global], [model.f_expl_expr])

    # replace groups in constraint expressions
    for k in range(model.con_h_expr.size1()):
        model.con_h_expr[k] = model.con_h_expr_e[k] = substitute_expression_through_sympy(
                casadi_expr=model.con_h_expr[k],
                casadi_vars=casadi_vars,
                sympy_dict=sympy_dict,
                substitutions=substitutions
            )

    # substitute car parameters if numerical values are given
    if car_params is not None:
        pi_groups_numeric = get_pi_groups(car_params)
        model.substitute(pi_groups_casadi, pi_groups_numeric)
        model.p_global = []  # symbolic parameters are not needed anymore

    return model


def substitute_expression_through_sympy(casadi_expr: ca.SX, casadi_vars: ca.SX, sympy_dict: dict, substitutions: dict) -> ca.SX:
    """Replaces expressions defined by 'substitutions'; have to use SymPy for proper symbolic manipulations."""
    expr_sympy, _ = casadi2sympy(expr=casadi_expr, sym_map=sympy_dict)
    expr_sympy_expanded = sp.expand(expr_sympy)
    expr_sympy_modified = sp.simplify(expr_sympy_expanded.subs(substitutions))
    expr_casadi_modified = sympy2casadi(expr_sympy_modified, sympy_dict.values(), casadi_vars)

    return expr_casadi_modified


# def test_integrator_casadi(car_params: CarParams, model: AcadosModel):
#     """Simulate the symbolic dynamics using CasADi."""
#     ode = {'x': model.x, 'ode': model.f_expl_expr, 'p': model.u}
#     integrator = ca.integrator('integrator', 'rk', ode, 0, car_params.dt.item())
#     x = [0.0, 0.0, 0.0, 0.0]
#     for _ in range(10):
#         res = integrator(x0=x, p=[1.0, 0.0])  # straight driving, full throttle
#         x = res['xf'].full().flatten()
#     print("="*20)
#     print("car length: ", car_params.l.item())
#     print("x final: ", x)
#     print("x scaled: ", x / car_params.l.item())
#     print("="*20)
#     return


# def test_dimensionless_integrator_casadi():
#     from tqdm import tqdm
#     """Formulate a symbolic dimensionless model and compare the results using random values."""
#     model_intermediate = car_model_ocp(car_params=SymbolicCarParams(), dimensionless=True)
#     model = replace_pi_groups(model_intermediate)

#     dt = get_default_car_params().dt[0] / get_transformation_matrices(get_default_car_params())[2][0]
#     ode_dict = {'x': model.x, 'p': ca.vcat([model.u, model.p_global]), 'ode': model.f_expl_expr}
#     integrator = ca.integrator('F', 'cvodes', ode_dict, 0.0, dt)

#     pi_groups_default = get_pi_groups(get_default_car_params())
#     pi_groups_large = get_pi_groups(get_large_car_params())

#     nx = model.x.size1()
#     nu = model.u.size1()

#     for _ in tqdm(range(int(1e4)), desc="Testing dimensionless integrator with random values"):
#         x = np.random.rand(nx)
#         u = np.random.rand(nu)
#         results_default = integrator(x0=x, p=np.concatenate([u, pi_groups_default]))["xf"]
#         results_large = integrator(x0=x, p=np.concatenate([u, pi_groups_large]))["xf"]
#         assert np.allclose(results_default, results_large)
    
#     print("dimensionless model checks out")


# def car_model_sim_dimensionless(car_params: CarParams) -> AcadosModel:
#     """Creates a purely dimensionless simulation model."""
#     ocp_model_intermediate = car_model_ocp(car_params=SymbolicCarParams(), dimensionless=True)
#     model_ocp = replace_pi_groups(ocp_model_intermediate, car_params)

#     # remove the control input rate from the model for simulation
#     nx = 4
#     nu = 2
#     model = AcadosModel()
#     model.f_impl_expr = model_ocp.f_impl_expr[0:nx]
#     model.f_expl_expr = model_ocp.f_expl_expr[0:nx]
#     model.x = model_ocp.x[0:nx]
#     model.xdot = model_ocp.xdot[0:nx]
#     model.u = model_ocp.x[nx:nx+nu]
#     model.name = model_ocp.name.replace("ocp", "sim")
#     model.x_labels = model_ocp.x_labels[0:nx]
#     model.u_labels = model_ocp.x_labels[nx:nx+nu]
#     model.t_label = model_ocp.t_label

#     return model


# def export_acados_integrator_dimensionless(car_params: CarParams) -> AcadosSimSolver:
#     """Creates an integrator with a purely dimensionless model."""
#     acados_sim = AcadosSim()
#     acados_sim.model = car_model_sim_dimensionless(car_params)
#     acados_sim.solver_options.T = (get_default_car_params().dt[0] / get_transformation_matrices(get_default_car_params())[2][0]).item()
#     acados_sim.solver_options.integrator_type = "ERK"
#     acados_sim.solver_options.num_stages = 4
#     acados_sim.solver_options.num_steps = 3

#     # assign a random directory to prevent overwriting
#     export_directory = str(Path(mkdtemp(prefix="jkh_acados_")))
#     acados_sim.code_gen_opts.code_export_directory = os.path.join(export_directory, "codegen")
#     json_file = os.path.join(export_directory, "sim_model.json")
#     # export_directory = os.path.join("codegen", f"sim_{car_params.l.item():.3g}".replace(".", "_"))
#     # json_file = os.path.join("json", f"sim_{car_params.l.item():.3g}".replace(".", "_") + ".json")    
#     print("Setting up acados integrator...")

#     acados_sim_solver = AcadosSimSolver(
#         acados_sim=acados_sim, 
#         verbose=False,
#         json_file=json_file,
#     )
    
#     # schedule folder deletion after the program finishes
#     DeleteDirectoryHook(acados_sim_solver, export_directory)

#     return acados_sim_solver


# from race_car.utils.scaling import compare_casadi_expressions
# model_1 = car_model_sim(car_params=get_default_car_params(), dimensionless=True)
# model_2 = car_model_sim_dimensionless(car_params=get_default_car_params())
# for k in range(model_1.f_expl_expr.size1()):
#     expr_match = compare_casadi_expressions(expr1=model_1.f_expl_expr[k], syms1=ca.vertcat(model_1.x, model_1.u),
#                                             expr2=model_2.f_expl_expr[k], syms2=ca.vertcat(model_2.x, model_2.u))
#     print(f"dynamics for state {k} ", "match" if expr_match else "do not match")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from race_car.utils.config import get_default_car_params, get_large_car_params
    from race_car.utils.scaling import get_transformation_matrices

    # 0: small dimensional vs. dimensionless
    # 1: large dimensional vs. dimensionless
    # 2: small vs. large (both dimensionless)
    test_type = 0

    match test_type:
        case 0:
            integrator_ref = export_acados_integrator(car_params=get_default_car_params(), dimensionless=False)
            integrator_sim = export_acados_integrator(car_params=get_default_car_params(), dimensionless=True)
            labels = ["dimensional", "dimensionless"]
            # formulations differ, define scaling matrices
            Mx_ref = Mx_inv_ref = np.eye(4)
            Mx_sim = get_transformation_matrices(get_default_car_params())[0][:4,:4]
            Mx_inv_sim = np.linalg.inv(Mx_sim)
        case 1:
            integrator_ref = export_acados_integrator(car_params=get_large_car_params(), dimensionless=False)
            integrator_sim = export_acados_integrator(car_params=get_large_car_params(), dimensionless=True)
            labels = ["dimensional", "dimensionless"]
            # formulations differ, define scaling matrices
            Mx_ref = Mx_inv_ref = np.eye(4)
            Mx_sim = get_transformation_matrices(get_large_car_params())[0][:4,:4]
            Mx_inv_sim = np.linalg.inv(Mx_sim)
        case 2:
            integrator_ref = export_acados_integrator(car_params=get_default_car_params(), dimensionless=True)
            integrator_sim = export_acados_integrator(car_params=get_large_car_params(), dimensionless=True)
            labels = ["small", "large"]
            # formulations are same, no transformation needed
            Mx_ref = Mx_inv_ref = Mx_sim = Mx_inv_sim = np.eye(4)
        # case 3:
        #     integrator_ref = export_acados_integrator(car_params=get_default_car_params(), dimensionless=True)
        #     integrator_sim = export_acados_integrator_dimensionless(car_params=get_default_car_params())
        #     labels = ["without subs", "with subs"]
        #     # formulations are same, no transformation needed
        #     Mx_ref = Mx_inv_ref = Mx_sim = Mx_inv_sim = np.eye(4)
        # case 4:
        #     integrator_ref = export_acados_integrator_dimensionless(car_params=get_default_car_params())
        #     integrator_sim = export_acados_integrator_dimensionless(car_params=get_large_car_params())
        #     labels = ["small", "large"]
        #     Mx_ref = Mx_inv_ref = Mx_sim = Mx_inv_sim = np.eye(4)
        case _:
            raise ValueError("test case not defined")
    print("Integrators successfully created.")

    # compare the integrators
    x0_ref = np.array([0.0, 0.0, 0.0, 0.0])
    x0_sim = x0_ref
    x_log_ref = []
    x_log_sim = []
    u_log = []

    for i in range(100):
        # choose a random input (physical, but unitless)
        u_rand = np.random.uniform(low=[-1.0, -0.4], high=[+1.0, +0.4])
        u_log.append(u_rand)

        # simulate the first system
        x_next_ref = integrator_ref.simulate(x=Mx_inv_ref @ x0_ref, u=u_rand)
        x_log_ref.append(Mx_ref @ x_next_ref)
        x0_ref = Mx_ref @ x_next_ref

        # simulate the second system
        x_next_sim = integrator_sim.simulate(x=Mx_inv_sim @ x0_sim, u=u_rand)
        x_log_sim.append(Mx_sim @ x_next_sim)
        x0_sim = Mx_sim @ x_next_sim

    print("Simulation successful.")
    print("Final state ref: ", x_log_ref[-1])
    print("Final state sim: ", x_log_sim[-1])

    # plot the results
    obs_ref_log = np.array(x_log_ref)
    obs_sim_log = np.array(x_log_sim)
    act_log = np.array(u_log)
    nx = 4
    nu = 2
    fig, ax = plt.subplots(nx + nu, 1, sharex=True)

    for i in range(nx):
        ax[i].plot(obs_ref_log[:, i], color="b", label=labels[0])
        ax[i].plot(obs_sim_log[:, i], color="r", linestyle="--", label=labels[1])
        ax[i].grid()
        ax[i].set_ylabel(integrator_ref.acados_sim.model.x_labels[i])
    ax[0].legend()

    for i in range(nu):
        ax[nx+i].step(
            list(range(obs_ref_log.shape[0]+1)),
            np.append([act_log[0, i]], act_log[:, i]),
            where="post",
            color="b",
        )
        ax[nx+i].set_ylabel(integrator_ref.acados_sim.model.u_labels[i])
        ax[nx+i].grid()
    ax[-1].set_xlabel("$k$")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels(ax)
    plt.show(block=False)

    if np.allclose(x_log_ref, x_log_sim):
        print("Results match.")
    else:
        print(f"Results do not match (max diff: {np.max(np.abs(np.array(x_log_ref)-np.array(x_log_sim))):.4f}).")

    print("Press ENTER to close the plot")
    input()
    plt.close()
