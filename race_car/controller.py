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


class RaceCarController(AcadosPlanner):

    def __init__(self, car_params: CarParams, dimensionless: bool):
        self.dimensionless = dimensionless

        # if not dimensionless:
        #     Mx, Mu, _ = get_transformation_matrices(car_params)
        #     Mx_tensor = torch.tensor(Mx, dtype=torch.float32)
        #     self.nondim2dim_x = lambda x: (x @ Mx_tensor.to(x.device))
        #     Mu_inv_tensor = torch.tensor(np.linalg.inv(Mu), dtype=torch.float32)
        #     self.dim2nondim_u = lambda u: (u @ Mu_inv_tensor.to(u.device))

        self.sref_N = 46.5 * (car_params.l.item() if not dimensionless else 1.0)  # terminal progress reference
        self.N = car_params.N.item()
        self.s_offset = np.linspace(start=0.0, stop=self.sref_N, num=self.N+1)  # will be added to current s

        param_manager = AcadosParameterManager(
            parameters=create_acados_params(car_params, dimensionless=dimensionless),
            N_horizon=car_params.N.item(),
        )

        self.ocp = export_parametric_ocp(
            car_params=car_params, dimensionless=dimensionless, param_manager=param_manager)

        print("Setting up acados MPC...")
        diff_mpc = AcadosDiffMpcTorch(
            ocp=self.ocp,
            # export_directory=self.ocp.code_gen_opts.code_export_directory,  # don't specify this to avoid issues with code reuse
            verbose=False
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)


    def forward(
        self, obs: torch.Tensor, param: torch.Tensor, ctx: AcadosDiffMpcCtx | None = None
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor]:
        batch_size = obs.shape[0]

        # if not self.dimensionless:
        #     obs = self.nondim2dim_x(obs)  # observations are always dimensionless

        s_current = obs[:, 0].cpu().numpy()          # (B,)
        sref = s_current[:, None] + self.s_offset    # (B, N+1)

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            sref=sref.reshape(batch_size, -1, 1),
        )

        ctx_out, u0, x, u, v = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )

        # TODO: remove printing
        # if ctx_out.status.item():
        #     print(f"MPC solver failed with status {ctx_out.status.item()}")
        #     lbx = self.diff_mpc.diff_mpc_fun.ocp.constraints.lbx[1:]
        #     ubx = self.diff_mpc.diff_mpc_fun.ocp.constraints.ubx[1:]
        #     u_pred = x[0,:,4:].numpy()

        #     violated_lbx = u_pred <= lbx
        #     violated_ubx = u_pred >= ubx
        #     for b in range(2):
        #         stages = np.where(violated_lbx[:, b])[0]
        #         if len(stages) > 0:
        #             stages_str = ", ".join(map(str, stages))
        #             print(f"input {b + 1} constraint violated at stages {{{stages_str}}}")
        #         stages = np.where(violated_ubx[:, b])[0]
        #         if len(stages) > 0:
        #             stages_str = ", ".join(map(str, stages))
        #             print(f"input {b + 1} constraint violated at stages {{{stages_str}}}")

        # if not self.dimensionless:
        #     u0 = self.dim2nondim_u(u0)  # actions should be dimensionless

        return ctx_out, u0


def export_parametric_ocp(car_params: CarParams, dimensionless: bool, param_manager: AcadosParameterManager) -> AcadosOcp:
    """Modifies the OCP for use within leap-c."""
    ocp = export_ocp(car_params=car_params, dimensionless=dimensionless)
    
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
        n_steps=ocp.solver_options.sim_method_num_steps
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