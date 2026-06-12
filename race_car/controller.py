"""Creates an MPC controller for use in leap-c."""
import casadi as ca
import numpy as np
import torch

from acados_template import AcadosOcp
from leap_c.examples.utils.casadi import integrate_erk4
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

from race_car.acados_ocp import export_ocp
from race_car.utils.config import CarParams, create_acados_params, get_default_car_params_tracking

# from acados_template import AcadosOcp, AcadosCasadiOcpSolver
# from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate
# from acados_template.utils import status_to_str
# resolve_with_casadi = False  # whether to try resolving with CasADi if the acados solver fails

class RaceCarController(AcadosPlanner):

    def __init__(self, car_params: CarParams, dimensionless: bool):
        self.dimensionless = dimensionless

        self.sref_N = car_params.s_ref_N.item() / (car_params.l.item() if dimensionless else 1.0)  # terminal progress reference
        self.N = car_params.N.item()
        self.s_offset = np.linspace(start=0.0, stop=self.sref_N, num=self.N+1)  # will be added to current s

        param_manager = AcadosParameterManager(
            parameters=create_acados_params(car_params, dimensionless=dimensionless),
            N_horizon=car_params.N.item(),
        )

        self.ocp = export_parametric_ocp(
            car_params=car_params, dimensionless=dimensionless, param_manager=param_manager, soften_constraints=True, promote_bounds=False
        )

        # if resolve_with_casadi:
        #     self.ocp = export_parametric_ocp(
        #         car_params=car_params, dimensionless=dimensionless, param_manager=param_manager, soften_constraints=True, promote_bounds=True
        #     )
        #     self.nonparametric_ocp = export_ocp(
        #         car_params=car_params, dimensionless=dimensionless, soften_constraints=True, promote_bounds=True
        #     )
        #     # test reformulation using acados -> CasADi
        #     casadi_solver = AcadosCasadiOcpSolver(self.nonparametric_ocp)

        print("Setting up acados MPC...")
        diff_mpc = AcadosDiffMpcTorch(
            ocp=self.ocp,
            # export_directory=self.ocp.code_gen_opts.code_export_directory,  # don't specify this to avoid issues with code reuse
            verbose=False,
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

        self.solver = self.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
        self.max_hessian_qp_cond = 0
        self.car_params = car_params


    def forward(
        self, obs: torch.Tensor, param: torch.Tensor, ctx: AcadosDiffMpcCtx | None = None
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor]:

        s_current = obs[:, 0].cpu().numpy()          # (B,)
        sref = s_current[:, None] + self.s_offset    # (B, N+1)

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            sref=sref.reshape(obs.shape[0], -1, 1),
        )

        ctx_out, u0, x, u, v = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )

        # TODO: remove debugging code
        """
        if ctx_out.status.item() and False:

            print("MPC solver failed: ", status_to_str(ctx_out.status.item()))

            ## print statistics and get last iterate
            if False:
                self.solver.print_statistics()
                a = self.solver.get_iterate()
                x_traj = np.array([a.x[k] for k in range(51)])
                u_traj = np.array([a.u[k] for k in range(50)])
                self.plot_mpc_trajectory(x, u)

            ## check the Hessian of the QP subproblems
            # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_solver.AcadosOcpSolver.get_hessian_block
            if False:
                hessian_conds = np.array([np.linalg.cond(self.solver.get_hessian_block(k)) for k in range(self.solver.N + 1)])
                max_hessian = np.max(hessian_conds, initial=0.0, where=np.isfinite(hessian_conds))
                max_hessian_stage = np.where(hessian_conds == max_hessian)[0][0]
                print(f"Maximum QP Hessian condition number: {max_hessian:.2e} at stage {max_hessian_stage}")

            ## check the condition number of the cost matrices
            if False:
                p = 10 ** self.solver.get_flat("p_global")
                Q = np.diag(p[:4])
                Qe = np.diag(p[4:])
                print("condition number of Q:", np.linalg.cond(Q))
                print("condition number of Qe:", np.linalg.cond(Qe))

            ## use AcadosCostConstraintEvaluator
            if False:
                from acados_template.mpc_utils import AcadosCostConstraintEvaluator
                e = AcadosCostConstraintEvaluator(self.ocp)
                e.update_all(self.solver)
                X = x.squeeze().cpu().numpy()
                U = u.squeeze().cpu().numpy()
                eval_results = e.evaluate(x=X[0,:], u=U[0,:])
                eval_results["violation_hard_constraints"]
            
            ## check QP diagnostics
            if False:
                full_hessian_diagnostics = self.solver.qp_diagnostics("FULL_HESSIAN")
                projected_hessian_diagnostics = self.solver.qp_diagnostics("PROJECTED_HESSIAN")
                min_eig_full = full_hessian_diagnostics['min_eigv_global']
                min_abs_eig_full = full_hessian_diagnostics['min_abs_eigv_global']
                min_abs_eig_proj_hess= projected_hessian_diagnostics['min_abs_eigv_global']
                min_eig_proj_hess = projected_hessian_diagnostics['min_eigv_global']
                min_eig_P = projected_hessian_diagnostics['min_eigv_P_global']
                min_abs_eig_P = projected_hessian_diagnostics['min_abs_eigv_P_global']

            if resolve_with_casadi:
                ctx_out, u0, x, u, v = self.try_resolve(obs, param, p_stagewise, ctx=ctx_out)
                if False:
                    self.plot_mpc_trajectory(x, u)

                if ctx_out.status.item():
                    print(f"Resolving with acados failed with status {status_to_str(ctx_out.status.item())}")
                else:
                    print("Resolving with acados succeeded")


            ## check input constraint violations
            if False:
                lbx = self.diff_mpc.diff_mpc_fun.ocp.constraints.lbx[1:]
                ubx = self.diff_mpc.diff_mpc_fun.ocp.constraints.ubx[1:]
                u_pred = x[0,:,4:].numpy()

                violated_lbx = u_pred <= lbx
                violated_ubx = u_pred >= ubx
                for b in range(2):
                    stages = np.where(violated_lbx[:, b])[0]
                    if len(stages) > 0:
                        stages_str = ", ".join(map(str, stages))
                        print(f"input {b + 1} lower bound violated at stages {{{stages_str}}}")
                    stages = np.where(violated_ubx[:, b])[0]
                    if len(stages) > 0:
                        stages_str = ", ".join(map(str, stages))
                        print(f"input {b + 1} upper bound violated at stages {{{stages_str}}}")
        """

        return ctx_out, u0

    """
    def try_resolve(self, obs: torch.Tensor, param: torch.Tensor, p_stagewise: np.ndarray, ctx: AcadosDiffMpcCtx | None = None):
        # set the cost weights and the progress reference
        nx = 4
        for k in range(nx):
            self.nonparametric_ocp.cost.W[k, k] = 10 ** param[0][k].item()
            self.nonparametric_ocp.cost.W_e[k, k] = 10 ** param[0][k+nx].item()

        casadi_solver = AcadosCasadiOcpSolver(self.nonparametric_ocp)
        for k in range(1, self.N+1):
            yref = np.zeros_like(self.nonparametric_ocp.cost.yref if k < self.N else self.nonparametric_ocp.cost.yref_e)
            yref[0] = p_stagewise[0][k].item()
            casadi_solver.set(k, "yref", yref)

        # set the initial state
        casadi_solver.set(0, 'lbx', obs.numpy()[0])
        casadi_solver.set(0, 'ubx', obs.numpy()[0])
        
        # solve the problem with CasADi
        print("Solving the same problem with IPOPT...")
        status = casadi_solver.solve()
        if status == "Solve_Succeeded":
            print("Resolving with IPOPT succeeded")
        else:
            print(f"Resolving with IPOPT failed with status {status}")
        
        # update the context with the last CasADi iterate and solve with acados again
        casadi_iterate = casadi_solver.get_flat_iterate()
        batch_size = obs.shape[0]
        # slack_iterate = np.zeros((batch_size, self.diff_mpc.diff_mpc_fun.ocp.dims.ns * self.N))  # set slack iterates to zero, not implemented yet
        ctx.iterate = AcadosOcpFlattenedBatchIterate(
            x=casadi_iterate.x.reshape(batch_size,-1),
            u=casadi_iterate.u.reshape(batch_size,-1),
            z=casadi_iterate.z.reshape(batch_size,-1),
            sl=casadi_iterate.sl.reshape(batch_size,-1),
            su=casadi_iterate.su.reshape(batch_size,-1),
            pi=casadi_iterate.pi.reshape(batch_size,-1),
            lam=casadi_iterate.lam.reshape(batch_size,-1),
            N_batch=batch_size,
        )

        # solver = self.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
        # solver.options_set("globalization_fixed_step_length", 0.5)

        print("Using IPOPT solution as initial guess for acados...")

        return self.diff_mpc(obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx)
    
    def plot_mpc_trajectory(self, x: torch.Tensor, u: torch.Tensor):
        from race_car.utils.plotting import plot_results_classic
        plot_results_classic(x.squeeze().numpy()[:self.N], u.squeeze().numpy(), list(range(self.N)), car_params=self.car_params)
    """

    def jacobian_action_param(self, ctx: AcadosDiffMpcCtx) -> np.ndarray:
        return self.sensitivity(ctx, "du0_dp")


def export_parametric_ocp(car_params: CarParams, dimensionless: bool, param_manager: AcadosParameterManager, soften_constraints: bool, promote_bounds: bool) -> AcadosOcp:
    """Modifies the OCP for use within leap-c."""
    ocp = export_ocp(car_params, dimensionless, soften_constraints, promote_bounds)
    
    p = ca.vertcat(
        param_manager.non_learnable_parameters.cat,
        param_manager.learnable_parameters.cat,
    )

    # define discrete dynamics (required for leap-c)
    ocp.model.disc_dyn_expr = integrate_erk4(
        f_expl=ocp.model.f_expl_expr,
        x=ocp.model.x,
        u=ocp.model.u,
        p=p,
        dt=ocp.solver_options.tf / ocp.solver_options.N_horizon,
        n_steps=ocp.solver_options.sim_method_num_steps,
    )
    ocp.solver_options.integrator_type = "DISCRETE"

    # adapt the cost to use AcadosParameters
    qx_diag_log = param_manager.get("qx_diag_log")
    qu_diag_log = param_manager.get("qu_diag_log")
    r_diag_log = param_manager.get("r_diag_log")
    qex_diag_log = param_manager.get("qex_diag_log")
    qeu_diag_log = param_manager.get("qeu_diag_log")

    # if not dimensionless:
    #     # scale the state cost weights to the specified car
    #     nx = 4
    #     Mx = get_transformation_matrices(car_params)[0][:nx, :nx]
    #     log_scale_correction = -2 * np.log10(np.diag(Mx))
    #     qx_diag_log += log_scale_correction
    #     qex_diag_log += log_scale_correction

    # set the NONLINEAR_LS cost
    ocp.cost.W_0 = ca.diag(10 ** (r_diag_log))
    ocp.cost.W = ca.diag(10 ** ca.vertcat(qx_diag_log, qu_diag_log, r_diag_log))
    ocp.cost.W_e = ca.diag(10 ** ca.vertcat(qex_diag_log, qeu_diag_log))

    sref = param_manager.get("sref")
    yref = ca.vertcat(sref, np.zeros(7))
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref[:-2]  # control inputs are not included in the terminal stage

    param_manager.assign_to_ocp(ocp)

    return ocp


if __name__ == "__main__":
    from race_car.utils.config import get_large_car_params_tracking
    from race_car.utils.scaling import compare_casadi_expressions

    # compare the dimensionless controllers
    a = RaceCarController(car_params=get_default_car_params_tracking(), dimensionless=True)
    b = RaceCarController(car_params=get_large_car_params_tracking(), dimensionless=True)
    c = a.diff_mpc.diff_mpc_fun.ocp.model
    d = b.diff_mpc.diff_mpc_fun.ocp.model
    syms_a = ca.vertcat(c.x, c.u, c.p, c.p_global, c.xdot)
    syms_b = ca.vertcat(d.x, d.u, d.p, d.p_global, d.xdot)

    assert np.allclose(a.param_manager.combine_default_learnable_parameter_values(), b.param_manager.combine_default_learnable_parameter_values())

    assert str(c.cost_expr_ext_cost_0) == str(d.cost_expr_ext_cost_0)
    assert str(c.cost_expr_ext_cost) == str(d.cost_expr_ext_cost)
    assert str(c.cost_expr_ext_cost_e) == str(d.cost_expr_ext_cost_e)

    assert compare_casadi_expressions(c.con_h_expr, syms_a, d.con_h_expr, syms_b)
    assert compare_casadi_expressions(c.con_h_expr_e, syms_a, d.con_h_expr_e, syms_b)
    assert compare_casadi_expressions(c.cost_expr_ext_cost_0, syms_a, d.cost_expr_ext_cost_0, syms_b)
    assert compare_casadi_expressions(c.disc_dyn_expr, syms_a, d.disc_dyn_expr, syms_b)
    assert compare_casadi_expressions(c.f_expl_expr, syms_a, d.f_expl_expr, syms_b)
    assert compare_casadi_expressions(c.f_impl_expr, syms_a, d.f_impl_expr, syms_b)

    # TODO: check bounds, sampling time etc. (move into utility function)

    print("ok")