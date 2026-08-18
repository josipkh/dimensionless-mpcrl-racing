"""Compares the analytic and empirical gradients of the closed-loop cost with respect to the dimensionless groups."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from race_car.utils.config import CarParams
from race_car.utils.scaling import get_pi_groups

from sensitivity_analysis.closed_loop_performance.analytical import simulate_closed_loop, calculate_closed_loop_cost_sensitivity
from sensitivity_analysis.closed_loop_performance.empirical import evaluate_closed_loop


def compare_gradients(car_params: CarParams, delta: float = 0.05, n_points: int = 7):

    pi_nominal = get_pi_groups(car_params)
    npi = len(pi_nominal)
    data_path = "output/gradient_comparison.npz"
    plot_output_path = "output/gradient_comparison.png"

    # load or generate the data
    os.makedirs("output", exist_ok=True)
    if os.path.exists(data_path):
        print(f"Loading saved data from {data_path}...")
        data = np.load(data_path)
        rel_perturbations = data["rel_perturbations"]
        delta_J_empirical_all = data["delta_J_empirical_all"]
        delta_J_analytic_all = data["delta_J_analytic_all"]
    else:
        print("Calculating analytical gradients...")
        log, baseline_cost = simulate_closed_loop(car_params)
        dJ_dp = calculate_closed_loop_cost_sensitivity(car_params, log)
        dJ_dp = np.asarray(dJ_dp).flatten()

        rel_perturbations = np.linspace(-delta, delta, n_points)
        delta_J_empirical_all = np.zeros((npi, n_points))
        delta_J_analytic_all = np.zeros((npi, n_points))

        print("Evaluating empirical points...")
        for i_param in range(npi):
            
            for idx, rel_p in enumerate(rel_perturbations):
                pi_vec = np.copy(pi_nominal)
                pi_vec[i_param] *= 1.0 + rel_p
                
                _, J_perturbed = evaluate_closed_loop(car_params, pi_vec)
                delta_J_empirical_all[i_param, idx] = J_perturbed - baseline_cost

            delta_J_analytic_all[i_param] = dJ_dp[i_param] * (rel_perturbations)
        delta_J_empirical_all /= baseline_cost  # normalize to get relative changes

        np.savez(
            data_path,
            rel_perturbations=rel_perturbations,
            delta_J_empirical_all=delta_J_empirical_all,
            delta_J_analytic_all=delta_J_analytic_all
        )
        print(f"Saved gradient data to {data_path}")

    
    print("Plotting comparison...")
    fig = plt.figure(figsize=(12, 8))
    spec = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)

    grid_positions = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (1, slice(1, 3)),
        (1, slice(3, 5)),
    ]

    axes = [fig.add_subplot(spec[r, c]) for r, c in grid_positions]

    for i_param in range(npi):
        ax = axes[i_param]
        
        delta_J_empirical = delta_J_empirical_all[i_param]
        delta_J_analytic = delta_J_analytic_all[i_param]

        ax.plot(
            rel_perturbations, 
            delta_J_empirical, 
            'o', 
            label="Empirical cost difference",
            color="tab:blue"
        )
        ax.plot(
            rel_perturbations, 
            delta_J_analytic, 
            '--', 
            label="Analytic sensitivity",
            color="tab:red", 
            linewidth=2
        )

        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(f"$\\Delta\\pi_{{{i_param+1}}} / \\pi_{{{i_param+1}}}$")
        if i_param in [0, 3]:  # Leftmost plots of row 1 and row 2
            ax.set_ylabel("$\\Delta J / J_0$")
            
        ax.set_title(f"$\\pi_{{{i_param+1}}}$")
        ax.grid(True)

    plt.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="center left", 
        bbox_to_anchor=(0.0, 0.45), 
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_output_path}.")
    plt.show()


if __name__ == "__main__":
    from race_car.utils.config import get_default_car_params
    car_params = get_default_car_params()
    compare_gradients(car_params, delta=0.05, n_points=7)