"""Test the importance of specified Pi-groups for dynamic similarity, through perturbation and one-step simulations."""
import os
import numpy as np
import casadi as ca
from scipy.stats import qmc
from tqdm import tqdm
import matplotlib.pyplot as plt

from acados_template import AcadosModel

from race_car.utils.config import SymbolicCarParams, get_default_car_params
from race_car.utils.scaling import get_pi_groups, get_transformation_matrices
from race_car.utils.track import get_track
from race_car.model import car_model_sim


def test_with_finite_differences(
        model: AcadosModel,
        dt: float,
        x_max: np.ndarray,
        x_min: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
        pi_nominal: np.ndarray,
        perturbation_indices: list,  # NOTE: 1-based
        output_file: str | None = None,
        N_samples: int = 500,  # global state and input coverage
        seed: int = 0,
) -> None:
    
    assert len(perturbation_indices) == 2, "evaluation with more than 2 perturbed groups not implemented yet"
    idx1 = perturbation_indices[0]
    idx2 = perturbation_indices[1]
    
    # create a CasADi integrator (one step)
    ode = {'x': model.x, 'p': ca.vertcat(model.u, model.p_global), 'ode': model.f_expl_expr}
    one_step_simulator = ca.integrator('one_step_sim', 'cvodes', ode, 0.0, dt)

    # sample initial states and controls
    l_bounds = np.hstack((x_min, u_min.flatten()))
    u_bounds = np.hstack((x_max, u_max.flatten()))

    sampler = qmc.LatinHypercube(d=l_bounds.shape[0], rng=seed)
    samples = sampler.random(n=N_samples)
    samples = qmc.scale(samples, l_bounds, u_bounds)
    
    nx = model.x.size1()
    X_samples = samples[:, :nx]
    U_samples = samples[:, nx:]

    # record nominal transitions
    x_next_nominal = np.zeros((N_samples, nx))    
    for i in range(N_samples):
        p_step = ca.vertcat(U_samples[i], pi_nominal)  # NOTE: Pi-groups kept at nominal values
        res = one_step_simulator(x0=X_samples[i], p=p_step)
        x_next_nominal[i, :] = np.array(res['xf']).flatten()

    # define the perturbation grid (and flatten)
    perturbation_range = np.linspace(-0.50, 0.50, 11)  # NOTE: change if needed
    dpi1_grid, dpi2_grid = np.meshgrid(perturbation_range, perturbation_range)
    dpi1_flat = dpi1_grid.flatten()
    dpi2_flat = dpi2_grid.flatten()
    N_grid_points = len(dpi1_flat)

    # evaluate perturbations and calculate mean squared error
    x_range = x_max - x_min
    mean_squared_errors = np.zeros(N_grid_points)

    for g in tqdm(range(N_grid_points), desc="Progress"):
        pi_perturbed = np.array(pi_nominal, copy=True)
        pi_perturbed[idx1 - 1] *= (1.0 + dpi1_flat[g])
        pi_perturbed[idx2 - 1] *= (1.0 + dpi2_flat[g])
        
        errors_at_grid_point = []
        
        for i in range(N_samples):
            p_step = ca.vertcat(U_samples[i], pi_perturbed)
            res = one_step_simulator(x0=X_samples[i], p=p_step)
            x_next_perturbed = np.array(res['xf']).flatten()
            
            # Range-normalized difference (Assuming x_range = x_max - x_min is fixed)
            raw_diff = x_next_perturbed - x_next_nominal[i, :]
            scaled_diff = raw_diff / x_range
            
            # square the norm to match the variance/DGSM scale of the active subspace method
            abs_error_squared = np.linalg.norm(scaled_diff) ** 2
            errors_at_grid_point.append(abs_error_squared)
            
        # Store the Mean Squared Error (MSE)
        mean_squared_errors[g] = np.mean(errors_at_grid_point)
    
    # normalize to the [0,1] range
    error_min, error_max = np.min(mean_squared_errors), np.max(mean_squared_errors)
    mean_squared_errors_normalized = (mean_squared_errors - error_min) / (error_max - error_min)

    if False:
        # scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            dpi1_flat, 
            dpi2_flat, 
            c=mean_relative_errors, 
            cmap='viridis', 
            s=150, 
            edgecolors='k', 
            alpha=0.85
        )        
        plt.plot(0, 0, 'rX', markersize=12, label=r'Nominal $\pi$')
        
        plt.title(r'System Sensitivity: Error in $x^+$ vs. Perturbations in $\pi_1$ & $\pi_2$', fontsize=14)
        plt.xlabel(r'Relative Perturbation in $\pi_1$ ($\Delta\pi_1 / \pi_1$)', fontsize=12)
        plt.ylabel(r'Relative Perturbation in $\pi_2$ ($\Delta\pi_2 / \pi_2$)', fontsize=12)
        cbar = plt.colorbar(scatter)
        cbar.set_label(r'Mean Relative State Error $\frac{\|| \Delta x^+ \||}{\|| x^+ \||}$', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

    # plot perturbations with color contours and isolines
    plt.figure(figsize=(11, 8.5))
    
    contour_filled = plt.tricontourf(
        dpi1_flat, 
        dpi2_flat, 
        mean_squared_errors_normalized, 
        levels=15, 
        cmap='viridis',
        alpha=0.85
    )
    
    plt.tricontour(
        dpi1_flat, 
        dpi2_flat, 
        mean_squared_errors_normalized, 
        levels=15, 
        colors='white', 
        linewidths=0.5,
        alpha=0.6
    )

    scatter = plt.scatter(
        dpi1_flat, 
        dpi2_flat, 
        c=mean_squared_errors_normalized, 
        cmap='viridis', 
        s=40,              
        edgecolors='black', 
        linewidths=0.5,
        zorder=3           
    )
    
    plt.plot(0, 0, 'rX', markersize=14, label=r'Nominal $\pi$', zorder=4)

    plt.xlabel(fr'Relative Perturbation in $\pi_{idx1}$ ($\Delta\pi_{idx1} / \pi_{idx1}$)', fontsize=12)
    plt.ylabel(fr'Relative Perturbation in $\pi_{idx2}$ ($\Delta\pi_{idx2} / \pi_{idx2}$)', fontsize=12)
    plt.title(fr'Mean Squared Error in $x^+$ vs. $\pi_{idx1}$ & $\pi_{idx2}$', fontsize=14, pad=15)
    
    cbar = plt.colorbar(contour_filled)
    cbar.set_label(r'$\mathbb{E}[||\Delta x^+||^2]$', fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.3, color='white')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # numerical analysis of the results
    pi1_axis_indices = np.where(dpi2_flat == 0)[0]
    pi2_axis_indices = np.where(dpi1_flat == 0)[0]

    # average squared slope
    pi1_sens_sq = np.mean([
        mean_squared_errors[idx] / (dpi1_flat[idx] ** 2)
        for idx in pi1_axis_indices if dpi1_flat[idx] != 0
    ])    
    pi2_sens_sq = np.mean([
        mean_squared_errors[idx] / (dpi2_flat[idx] ** 2)
        for idx in pi2_axis_indices if dpi2_flat[idx] != 0
    ])

    # estimate relative importance
    total_shared_importance = pi1_sens_sq + pi2_sens_sq
    pi1_share = (pi1_sens_sq / total_shared_importance) * 100
    pi2_share = (pi2_sens_sq / total_shared_importance) * 100

    print(f"Relative importance of pi_{idx1} (finite differences): {pi1_share:.2f} %")
    print(f"Relative importance of pi_{idx2} (finite differences): {pi2_share:.2f} %")
    # print(f"Ratio (pi_{idx1} / pi_{idx2}): {pi2_sens_sq / pi1_sens_sq:.1f}x")

    if "DISPLAY" in os.environ and output_file is None:
        plt.show()
    else:
        output_path = output_file or "output/plot_sens_similarity.png"
        plt.savefig(output_path, dpi=300)
        print(f"saved plot to {output_path}")
        plt.close()


if __name__ == "__main__":
    # model and settings
    model = car_model_sim(car_params=SymbolicCarParams(), dimensionless=True)    
    default_car_params = get_default_car_params()
    pi_nominal = get_pi_groups(default_car_params)
    perturbation_indices = [1, 2]  # NOTE: 1-based

    Mx, _, mt = get_transformation_matrices(default_car_params)
    dt = default_car_params.dt.item() / mt.item()
    
    # dimensionless state limits (s, n, alpha, v)
    x_max_dimensional = np.array([
        get_track(default_car_params)[0][-1],
        default_car_params.n_max.item(),
        np.pi,
        default_car_params.v_max.item(),
    ])
    x_min_dimensional = np.array([
        0.0,
        default_car_params.n_min.item(),
        -np.pi,
        0.0,
    ])
    Mx_inv = np.linalg.inv(Mx[:4, :4])
    x_max = Mx_inv @ x_max_dimensional
    x_min = Mx_inv @ x_min_dimensional
    
    # dimensionless input limits (D, delta; no scaling needed)
    u_max = np.array([default_car_params.D_max, default_car_params.delta_max])
    u_min = np.array([default_car_params.D_min, default_car_params.delta_min])

    test_with_finite_differences(model, dt, x_max, x_min, u_max, u_min, pi_nominal, perturbation_indices)
