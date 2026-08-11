"""Utility functions for plotting."""
import matplotlib
import os
if not "DISPLAY" in os.environ:
    matplotlib.use('Agg')  # Non-interactive backend

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from race_car.utils.config import CarParams, get_default_car_params
from race_car.utils.track import frenet2global, get_track, global2frenet


def plot_track(car_params: CarParams):
    """Plots the track for the given car parameters."""
    import matplotlib.pyplot as plt

    _, xref, yref, psiref, _ = get_track(car_params=car_params)

    plt.figure(figsize=(10, 8))
    plt.plot(xref, yref, label='Track Centerline', color='blue')
    step = 20
    plt.quiver(
        xref[::step],
        yref[::step],
        np.cos(psiref[::step]),
        np.sin(psiref[::step]),
        angles='xy',
        scale_units='xy',
        scale=5*get_default_car_params().l.item()/car_params.l.item(),
        color='red',
        width=0.003,
        label='Heading'
    )
    plt.axis('equal')
    plt.title('Track Centerline with Heading Arrows {}'.format('(L = {:.2f} m)'.format(car_params.l.item())))
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def plot_Fxd_surface(car_params: CarParams):
    """
    Plots the Fxd curve over speed (v) and throttle (D).

    Parameters:
        cm1, cm2, cr2, cr0, cr3: car parameters
        v_range: tuple (min_v, max_v)
        D_range: tuple (min_D, max_D)
        resolution: number of points for meshgrid
    """
    default_v_max = get_default_car_params().v_max.item()
    default_cr3 = get_default_car_params().cr3.item()
    v_max = default_v_max * default_cr3 / car_params.cr3.item()
    v_range = (0, v_max)
    D_range = (0, 1)
    resolution = 100

    # Create meshgrid of v and D
    v = np.linspace(*v_range, resolution)
    D = np.linspace(*D_range, resolution)
    V, D_grid = np.meshgrid(v, D)

    # Compute Fxd over the grid
    cm1 = car_params.cm1.item()
    cm2 = car_params.cm2.item()
    cr2 = car_params.cr2.item()
    cr0 = car_params.cr0.item()
    cr3 = car_params.cr3.item()
    Fxd = (cm1 - cm2 * V) * D_grid - cr2 * V**2 - cr0 * np.tanh(cr3 * V)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(V, D_grid, Fxd, cmap='viridis')
    ax.set_xlabel('Speed v [m/s]')
    ax.set_ylabel('Throttle D [0-1]')
    ax.set_zlabel('Fxd [N]')
    ax.set_title('Fxd vs Speed and Throttle (L = {:.2f} m)'.format(car_params.l.item()))
    plt.tight_layout()
    plt.show(block=False)


def plot_Fxd_vs_D_slices(car_params: CarParams, num_slices: int = 10):
    """
    Plots Fxd vs Throttle (D) for different fixed speeds (v).
    """
    default_v_max = get_default_car_params().v_max.item()
    default_cr3 = get_default_car_params().cr3.item()
    v_max = default_v_max * default_cr3 / car_params.cr3.item()

    cm1 = car_params.cm1.item()
    cm2 = car_params.cm2.item()
    cr2 = car_params.cr2.item()
    cr0 = car_params.cr0.item()
    cr3 = car_params.cr3.item()

    D = np.linspace(0, 1, 200)

    # Choose speeds for slicing (linearly spaced from 0 to v_max)
    v_slices = np.linspace(0, v_max, num_slices)

    plt.figure(figsize=(10, 6))

    for v in v_slices:
        Fxd = (cm1 - cm2 * v) * D - cr2 * v**2 - cr0 * np.tanh(cr3 * v)
        label = f"v = {v:.2f} m/s"
        plt.plot(D, Fxd, label=label)

    plt.xlabel("Throttle D [0-1]")
    plt.ylabel("Fxd [N]")
    plt.title("Fxd vs Throttle for Different Speeds (L = {:.2f} m)".format(car_params.l.item()))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def plot_results_track(
    state_log: np.ndarray,
    car_params: CarParams,
    total_time: float,
    output_file: str | None = None,
    preview_log: np.ndarray | None = None,
):
    """Plot the race track and the driven trajectory.
    
    Args:
        state_log:   Driven trajectory states, shape (Nsim, nx).
        car_params:  Car parameters.
        total_time:  Total lap time in seconds.
        output_file: Optional path to save the figure.
        preview_log: Optional MPC horizon preview, shape (N_horizon, nx).
                     If provided, the predicted trajectory is overlaid on the track.
    """
    # extract states from the log (shape Nsim x nx)
    s = state_log[:, 0]
    n = state_log[:, 1]
    alpha = state_log[:, 2]
    v = state_log[:, 3]

    default_distance = 0.12  # [m] half of track width for the small car
    distance = default_distance * car_params.l.item() / get_default_car_params().l.item()

    # transform projection to global coordinates
    [x, y, _, _] = frenet2global(s=s, n=n, alpha=alpha, v=v, car_params=car_params)
    [Sref, Xref, Yref, Psiref, _] = get_track(car_params)

    # setup plot
    ymin = np.min(Yref) - 5 * car_params.l.item()
    ymax = np.max(Yref) + 5 * car_params.l.item()
    xmin = np.min(Xref) - 5 * car_params.l.item()
    xmax = np.max(Xref) + 5 * car_params.l.item()

    plt.figure()
    plt.ylim(bottom=ymin, top=ymax)
    plt.xlim(left=xmin, right=xmax)
    plt.ylabel('$y$ [m]')
    plt.xlabel('$x$ [m]')
    if total_time:
        plt.title("Lap time: {0:.2f} s".format(total_time))

    # plot center line
    plt.plot(Xref, Yref, '--', color='k')

    # define the speed range
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=car_params.v_max)

    # draw track boundaries
    Xboundleft  = Xref - distance * np.sin(Psiref)
    Yboundleft  = Yref + distance * np.cos(Psiref)
    Xboundright = Xref + distance * np.sin(Psiref)
    Yboundright = Yref - distance * np.cos(Psiref)
    plt.plot(Xboundleft,  Yboundleft,  color='k', linewidth=1)
    plt.plot(Xboundright, Yboundright, color='k', linewidth=1)

    # draw driven trajectory
    plt.plot(x, y, '-b')
    heatmap = plt.scatter(x, y, c=v, cmap=cm.rainbow, norm=norm, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.ax.set_title("   $v_x$ [m/s]", y=1.025)

    # draw MPC preview if available
    if preview_log is not None:
        s_p     = preview_log[:, 0]
        n_p     = preview_log[:, 1]
        alpha_p = preview_log[:, 2]
        v_p     = preview_log[:, 3]

        [x_p, y_p, _, _] = frenet2global(s=s_p, n=n_p, alpha=alpha_p, v=v_p, car_params=car_params)

        plt.plot(x_p, y_p, '-', color='orange', linewidth=1.2,
                 alpha=0.7, zorder=3)  # label='MPC preview'

        plt.scatter(
            x_p, y_p,
            c=v_p,
            cmap=cm.rainbow,   # same colormap
            norm=norm,         # same norm
            edgecolor='none',
            marker='o',
            s=18,
            alpha=0.85,
            zorder=4,
        )

        # Mark the start and end of the horizon
        plt.plot(x_p[0],  y_p[0],  's', color='orange',
                 markersize=7, zorder=5, label='Horizon start')
        plt.plot(x_p[-1], y_p[-1], '*', color='red',
                 markersize=9, zorder=5, label='Horizon end')

        # plt.legend(loc='upper right', fontsize=8)

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # put markers for s values
    step_size = 15.5 * car_params.l.item()
    n_steps   = int(Sref[-1] / step_size) + 1
    xi  = np.zeros(n_steps)
    yi  = np.zeros(n_steps)
    xi1 = np.zeros(n_steps)
    yi1 = np.zeros(n_steps)
    xi2 = np.zeros(n_steps)
    yi2 = np.zeros(n_steps)

    for i in range(n_steps):
        try:
            k = list(Sref).index(i * step_size + min(abs(Sref - i * step_size)))
        except:
            k = list(Sref).index(i * step_size - min(abs(Sref - i * step_size)))

        _, nrefi, _, _ = global2frenet(Xref[k], Yref[k], Psiref[k], 0, car_params)
        xi[i],  yi[i],  _, _ = frenet2global(Sref[k], nrefi + 2.0 * distance,    0, 0, car_params)
        xi1[i], yi1[i], _, _ = frenet2global(Sref[k], nrefi + distance,           0, 0, car_params)
        xi2[i], yi2[i], _, _ = frenet2global(Sref[k], nrefi + 1.25 * distance,    0, 0, car_params)

        plt.text(xi[i], yi[i], '{} m'.format(int(i * step_size)),
                 fontsize=10, horizontalalignment='center', verticalalignment='center')
        plt.plot([xi1[i], xi2[i]], [yi1[i], yi2[i]], color='black')

    if "DISPLAY" in os.environ and output_file is None:
        plt.show(block=False)
    else:
        output_path = output_file or "output/plot_track.png"
        plt.savefig(output_path, dpi=300)
        print(f"saved track plot to {output_path}")
        plt.close()


def plot_results_classic(simX: np.ndarray, simU: np.ndarray, t: np.ndarray, car_params: CarParams, output_file: str | None = None):
    """Plot the states and inputs over time, in a classic way."""
    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(t, simX[:,:])
    plt.ylabel('$x$')
    plt.legend(['$s$', '$n$', r'$\alpha$', '$v$', '$D$', r'$\delta$'])
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(t, simU[:,0], color='r')
    plt.plot([t[0], t[-1]], [car_params.dD_max, car_params.dD_max], 'r--', alpha=0.5, label='_nolegend_')
    plt.plot([t[0], t[-1]], [car_params.dD_min, car_params.dD_min], 'r--', alpha=0.5, label='_nolegend_')
    plt.step(t, simU[:,1], color='g')
    plt.plot([t[0], t[-1]], [car_params.ddelta_max, car_params.ddelta_max], 'g--', alpha=0.5, label='_nolegend_')
    plt.plot([t[0], t[-1]], [car_params.ddelta_min, car_params.ddelta_min], 'g--', alpha=0.5, label='_nolegend_')
    plt.legend([r'$\Delta D$', r'$\Delta\delta$'])
    plt.ylabel('$u$')
    plt.xlabel('$t$ [s]')
    plt.grid(True)
    fig.align_ylabels()

    if "DISPLAY" in os.environ and output_file is None:
        plt.show(block=False)
    else:
        output_path = output_file or "output/plot_classic.png"
        plt.savefig(output_path)
        print(f"saved classic plot to {output_path}")
        plt.close()


def plot_lat_acc(simX: np.ndarray, t: np.ndarray, car_params: CarParams, output_file: str | None = None) -> None:
    """Plot the lateral acceleration over time, with constraints."""
    Nsim = t.shape[0]
    plt.figure()
    alat = np.zeros(Nsim)
    for i in range(Nsim):
        alat[i] = calculate_acc(simX[i,:], car_params)
    plt.plot(t, alat)
    plt.plot([t[0], t[-1]], [car_params.a_lat_min, car_params.a_lat_min], 'k--')
    plt.plot([t[0], t[-1]], [car_params.a_lat_max, car_params.a_lat_max], 'k--')
    plt.xlabel('$t$ [s]')
    plt.ylabel('$a_y$ [m/s$^2$]')
    plt.grid(True)

    if "DISPLAY" in os.environ and output_file is None:
        plt.show(block=False)
    else:
        output_path = output_file or "output/plot_lat_acc.png"
        plt.savefig(output_path)
        print(f"saved lateral acceleration plot to {output_path}")
        plt.close()


def calculate_acc(x: np.ndarray, car_params: CarParams) -> np.ndarray:
    m = car_params.m[0]
    c1 = car_params.lr[0] / car_params.l[0]
    c2 = 1 / car_params.l[0]
    cm1 = car_params.cm1[0]
    cm2 = car_params.cm2[0]
    cr0 = car_params.cr0[0]
    cr2 = car_params.cr2[0]
    cr3 = car_params.cr3[0]

    D = x[4]
    delta = x[5]
    v = x[3]
    Fxd = (cm1 - cm2 * v) * D - cr2 * v * v - cr0 * ca.tanh(cr3 * v)
    alat = c2 * v * v * delta + Fxd * ca.sin(c1 * delta) / m

    return alat


if __name__ == "__main__":
    from race_car.utils.config import get_large_car_params

    car_params = get_default_car_params()
    car_params_sim = get_large_car_params()

    # plot_track(car_params=car_params)
    # plot_track(car_params=car_params_sim)

    # plot_Fxd_surface(car_params=car_params)
    # plot_Fxd_surface(car_params=car_params_sim)

    # plot_Fxd_vs_D_slices(car_params=car_params)
    # plot_Fxd_vs_D_slices(car_params=car_params_sim)

    try:
        npz_file = np.load("race_car/jkh_test/small_car_results.npz")
        simX, simU, t = npz_file["simX"], npz_file["simU"], npz_file["t"]

        # test plotting with a partial log
        # trunc_idx = round(simX.shape[0]/2)
        # simX = simX[:trunc_idx, :]
        # simU = simU[:trunc_idx, :]
        # t = t[:trunc_idx]

        plot_results_track(simX, car_params, total_time=t[-1])
        plot_results_classic(simX, simU, t, car_params)
        plot_lat_acc(simX, t, car_params)
    except OSError:
        print("Specified log file could not be loaded, skipping result plots.")

    if plt.get_fignums():
        input("Press Enter to continue...")  # keep the plots open
        plt.close('all')

    print("ok")