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
from race_car.utils.scaling import get_transformation_matrices


class RaceCarController(AcadosPlanner):

    def __init__(self, car_params: CarParams, dimensionless: bool):
        self.car_params = car_params
        self.dimensionless = dimensionless

        # TODO: define nondim2dim here
        if dimensionless:
            Mx = get_transformation_matrices(car_params)[0]
            Mx_inv = np.linalg.inv(Mx)
            Mx_inv_tensor = torch.tensor(Mx_inv, dtype=torch.float32)
            self.dim2nondim_x = lambda x: (x @ Mx_inv_tensor.to(x.device))

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

        if self.dimensionless:
            # observations are dimensional, transform the state to the dimensionless space
            # TODO: always work with dimensionless observations, dimensionalize here if needed
            obs = self.dim2nondim_x(obs)

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
    qx_diag_log_default = param_manager.get("qx_diag_log")
    qu_diag_log = param_manager.get("qu_diag_log")
    r_diag_log = param_manager.get("r_diag_log")
    qex_diag_log_default = param_manager.get("qex_diag_log")
    qeu_diag_log = param_manager.get("qeu_diag_log")

    # start TODO: always work with dimensionless cost weights, just scale to the given car if needed
    # scale the state cost weights (due to the scale difference)
    nx = 4
    Mx = get_transformation_matrices(get_default_car_params_tracking())[0][:nx, :nx]
    mx = get_transformation_matrices(car_params)[0][:nx, :nx]
    # M = Mx @ np.linalg.inv(mx) is diagonal, so M.T @ diag(10^q) @ M = diag(10^(q + 2*log10(diag(M))))
    # also, log10(diag(M)) = log10(diag(Mx)) - log10(diag(mx))
    log_scale_correction = 2 * (np.log10(np.diag(Mx)) - np.log10(np.diag(mx)))

    qx_diag_log = qx_diag_log_default + log_scale_correction
    qex_diag_log = qex_diag_log_default + log_scale_correction

    # additional scaling if using the dimensionless formulation
    if dimensionless:
        log_scale_correction_dim = 2 * np.log10(np.diag(mx))
        qx_diag_log = qx_diag_log + log_scale_correction_dim
        qex_diag_log = qex_diag_log + log_scale_correction_dim

    # end TODO

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
    a = RaceCarController(car_params=get_default_car_params_tracking(), dimensionless=True)
    b = RaceCarController(car_params=get_large_car_params_tracking(), dimensionless=True)
    print("ok")