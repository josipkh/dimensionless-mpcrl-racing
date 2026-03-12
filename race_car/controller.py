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
from race_car.utils.config import CarParams, create_acados_params, get_default_car_params


class RaceCarController(AcadosPlanner):

    def __init__(self, car_params: CarParams, dimensionless: bool):
        self.car_params = car_params
        self.dimensionless = dimensionless

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
            export_directory=self.ocp.code_gen_opts.code_export_directory,
            verbose=False
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)


    def forward(
        self, obs: torch.Tensor, param: torch.Tensor, ctx: AcadosDiffMpcCtx | None = None
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor]:
        batch_size = obs.shape[0]

        s_current = np.array([obs[i, 0].cpu().numpy() for i in range(batch_size)])
        sref = s_current + self.s_offset

        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            sref=sref.reshape(batch_size, -1, 1),
        )

        ctx, u0, _, _, _ = self.diff_mpc(
            obs, p_global=param, p_stagewise=p_stagewise, ctx=ctx
        )
        return ctx, u0


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
    )
    ocp.solver_options.integrator_type = "DISCRETE"

    # adapt the cost to use AcadosParameters
    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    qe_diag_sqrt = param_manager.get("qe_diag_sqrt")

    # set the NONLINEAR_LS cost
    W_0_sqrt = ca.diag(r_diag_sqrt)
    ocp.cost.W_0 = W_0_sqrt @ W_0_sqrt.T

    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    ocp.cost.W = W_sqrt @ W_sqrt.T
    
    W_e_sqrt = ca.diag(qe_diag_sqrt)
    ocp.cost.W_e = W_e_sqrt @ W_e_sqrt.T

    sref = param_manager.get("sref")
    yref = ca.vertcat(sref, np.zeros(7))
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref[:-2]  # control inputs are not included in the terminal stage

    param_manager.assign_to_ocp(ocp)

    return ocp


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params
    RaceCarController(car_params=get_default_car_params(), dimensionless=True)
    print("ok")