"""Utility functions for plotting."""
import glob
import json
import os
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')  # Non-interactive backend

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

from race_car.utils.config import CarParams, get_small_car_params
from race_car.utils.track import frenet2global, get_track, global2frenet


TRANSFER_FORMULATIONS = ("dimensional", "dimensionless")
TRANSFER_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
TRANSFER_PLOT_NAME = "plot_transfer.pdf"


def _savefig(output_path: str | os.PathLike) -> None:
    """Save the current figure, creating its parent directory if necessary."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


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

    small_distance = 0.12  # [m] half of track width for the small car
    distance = small_distance * car_params.l.item() / get_small_car_params().l.item()

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
        _savefig(output_path)
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
        _savefig(output_path)
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
        _savefig(output_path)
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


def _load_transfer_manifest(folder: str | Path) -> dict | None:
    """Return the transfer.json of a run/batch folder, if it has one."""
    path = Path(folder) / "transfer.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _transfer_label(folder: str | Path) -> str:
    """Read the transfer direction from a run's manifest (or its legacy log)."""
    folder = Path(folder)
    manifest = _load_transfer_manifest(folder)
    if manifest and "label" in manifest:
        return manifest["label"]

    legacy_log = folder / "dimensional" / "seed_0" / "output.log"
    if legacy_log.is_file():
        lines = legacy_log.read_text().splitlines()
        if lines:
            return "large -> small" if "large car" in lines[0] else "small -> large"

    raise ValueError(
        f"Cannot determine transfer direction for {folder}: no transfer.json or output.log"
    )


def _extract_tensorboard_scalar(folder: str | Path) -> tuple[list[int], list[float]] | None:
    """Extract steps and values of the plotted scalar from a reduced TF folder."""
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(glob.glob(str(Path(folder) / "events.out.tfevents.*")))
    if not event_files:
        print(f"[plot_transfer] no TensorBoard files found in {folder}")
        return None, None

    event_file = event_files[0]
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[plot_transfer] no scalar metrics found in {event_file}")
        return None, None

    tag = tags[1] if len(tags) > 1 else tags[0]
    events = ea.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def _collect_transfer_curves(
    folder: str | Path,
    label: str | None = None,
    direction_key: str | None = None,
    scale_change: int | None = None,
) -> list[dict]:
    """Collect the two formulation curves of one direction from a run folder."""
    folder = Path(folder)
    manifest = _load_transfer_manifest(folder)

    manifest_direction = None
    manifest_label = None
    manifest_scale = None
    if manifest and "direction" in manifest and "directions" not in manifest:
        manifest_direction = manifest["direction"]
        manifest_label = manifest.get("label")
        manifest_scale = manifest.get("source_steps")

    curves = []
    for form in TRANSFER_FORMULATIONS:
        path = folder / form / "mean"
        if not path.is_dir():
            print(f"[plot_transfer] {path} not found, continuing...")
            continue

        if direction_key is None:
            direction_key = manifest_direction or label or _transfer_label(folder)
        if label is None:
            label = manifest_label or direction_key or _transfer_label(folder)
        if scale_change is None:
            scale_change = manifest_scale

        steps, values = _extract_tensorboard_scalar(path)
        if steps is None or values is None:
            continue
        if values[0] > 1:
            values = [value * 0.062 for value in values]

        curves.append({
            "path": str(path),
            "key": direction_key,
            "label": label,
            "form": form,
            "steps": steps,
            "values": values,
            # Legacy runs have no manifest: fall back to the midpoint assumption.
            "scale_change": scale_change if scale_change is not None else steps[-1] // 2,
        })
    return curves


def _collect_batch_transfer_curves(root: str | Path) -> list[dict] | None:
    """Collect all direction x formulation curves from a batch root."""
    manifest = _load_transfer_manifest(root)
    if not manifest or "directions" not in manifest:
        return None

    curves = []
    for direction, info in manifest["directions"].items():
        direction_manifest = _load_transfer_manifest(Path(root) / direction) or {}
        label = info.get("label") or direction_manifest.get("label") or direction
        scale_change = info.get("source_steps") or direction_manifest.get("source_steps")
        curves.extend(
            _collect_transfer_curves(
                Path(root) / direction,
                label=label,
                direction_key=direction,
                scale_change=scale_change,
            )
        )
    return curves


def collect_transfer_curves(runs: Sequence[str | Path]) -> list[dict]:
    """Accept one batch folder, one single run, or two legacy run folders."""
    runs = [Path(run) for run in runs]
    if len(runs) == 1:
        batch_curves = _collect_batch_transfer_curves(runs[0])
        return batch_curves if batch_curves is not None else _collect_transfer_curves(runs[0])
    if len(runs) == 2:
        return _collect_transfer_curves(runs[0]) + _collect_transfer_curves(runs[1])
    raise ValueError("Provide one batch folder or two run folders")


def find_transfer_batch_roots(folder: str | Path) -> list[Path]:
    """Find transfer batch roots under a batch root, a date folder, or a run folder."""
    folder = Path(folder)
    manifest = _load_transfer_manifest(folder)
    if manifest and "directions" in manifest:
        return [folder]

    nested = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        child_manifest = _load_transfer_manifest(child)
        if child_manifest and "directions" in child_manifest:
            nested.append(child)
    return nested or [folder]


def plot_transfer_curves(
    curves: list[dict],
    output_path: str | Path,
    joint_plot: bool = True,
) -> Path:
    """Plot direction x formulation curves; colour = direction, linestyle = formulation."""
    if not curves:
        raise ValueError("No transfer curves to plot")

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    colors = {}
    for curve in curves:
        if curve["key"] not in colors:
            colors[curve["key"]] = TRANSFER_COLORS[len(colors) % len(TRANSFER_COLORS)]
        print(f"[plot_transfer] loaded data from {curve['path']}")

    min_x = min(min(curve["steps"]) for curve in curves)
    max_x = max(max(curve["steps"]) for curve in curves)
    min_y = min(min(curve["values"]) for curve in curves)
    max_y = max(max(curve["values"]) for curve in curves)

    y_margin = (max_y - min_y) * 0.05
    fixed_xlim = (min_x, max_x)
    fixed_ylim = (min_y - y_margin, max_y + y_margin)
    scale_markers = sorted({int(curve["scale_change"]) for curve in curves if curve["scale_change"]})

    def draw_annotations(steps, values):
        for x in scale_markers:
            plt.axvline(x=x, color="k", lw=1, linestyle="--")
        y_baseline = -73.775 if values[0] < 0 else 281 * 0.062
        plt.hlines(y=y_baseline, xmin=0, xmax=steps[-1], color="k", lw=1, linestyle="--")
        plt.text(x=0, y=y_baseline, s="manual tuning", ha="left", va="bottom", fontsize=11, color="k")

        plt.xlim(fixed_xlim)
        plt.ylim(fixed_ylim)
        for x in scale_markers:
            plt.text(x=x, y=fixed_ylim[1] * 0.975, s=" scale change", ha="left", va="top",
                     fontsize=11, color="k")
        plt.xlabel("Training steps")
        plt.ylabel("Validation score" if values[0] < 0 else "Dimensionless racing time")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

    direction_handles = []
    seen = set()
    for curve in curves:
        if curve["key"] not in seen:
            seen.add(curve["key"])
            direction_handles.append(
                Line2D([0], [0], color=colors[curve["key"]], lw=2, label=curve["label"])
            )

    formulation_handles = [
        Line2D([0], [0], color="k", lw=2, linestyle="-", label="dimensionless"),
        Line2D([0], [0], color="k", lw=2, linestyle="--", label="dimensional"),
    ]

    output_path = Path(output_path)
    base_output, ext = os.path.splitext(output_path)

    if not joint_plot:
        plt.figure(figsize=(10, 6))
        dimensional = [curve for curve in curves if curve["form"] == "dimensional"]
        for curve in dimensional:
            plt.plot(curve["steps"], curve["values"], color=colors[curve["key"]],
                     linestyle="--", linewidth=2.5, alpha=1.0)

        if dimensional:
            last = dimensional[-1]
            draw_annotations(last["steps"], last["values"])
            plt.legend(handles=direction_handles, title="Transfer type", loc="upper left")
            dim_path = Path(f"{base_output}_dimensional{ext}")
            _savefig(dim_path)
            plt.close()
            print(f"[plot_transfer] saved the dimensional plot to {dim_path}")

    plt.figure(figsize=(10, 6))
    for curve in curves:
        linestyle = "--" if curve["form"] == "dimensional" else "-"
        alpha = 0.3 if not joint_plot and curve["form"] == "dimensional" else 1.0
        plt.plot(curve["steps"], curve["values"], color=colors[curve["key"]],
                 linestyle=linestyle, linewidth=2.5, alpha=alpha)

    draw_annotations(curves[-1]["steps"], curves[-1]["values"])
    first_legend = plt.legend(handles=direction_handles, title="Transfer type", loc="upper left")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=formulation_handles, title="Formulation", loc="upper left",
               bbox_to_anchor=(0.22, 1.0))

    final_path = output_path if joint_plot else Path(f"{base_output}_combined{ext}")
    _savefig(final_path)
    plt.close()
    print(f"[plot_transfer] saved the combined plot to {final_path}")
    return final_path


def plot_transfer(runs: Sequence[str | Path], output_path: str | Path | None = None) -> Path:
    """Plot one batch folder or two legacy run folders and return the saved PDF path."""
    curves = collect_transfer_curves(runs)
    if not curves:
        raise RuntimeError("No TensorBoard data found in the given folder(s)")

    runs = [Path(run) for run in runs]
    out = Path(output_path) if output_path is not None else runs[0] / TRANSFER_PLOT_NAME
    for key in dict.fromkeys(curve["key"] for curve in curves):
        label = next(curve["label"] for curve in curves if curve["key"] == key)
        print(f"[plot_transfer] {key}: {label}")
    return plot_transfer_curves(curves, out, joint_plot=True)


def plot_transfer_batches(folder: str | Path, output_name: str = TRANSFER_PLOT_NAME) -> list[Path]:
    """Plot every transfer batch under a batch root, date folder, or run folder."""
    outputs = []
    for root in find_transfer_batch_roots(folder):
        try:
            outputs.append(plot_transfer([root], output_path=root / output_name))
        except RuntimeError as exc:
            print(f"[plot_transfer] {exc}; skipping {root}")
    return outputs


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot transfer batches, or run the local smoke test.")
    parser.add_argument(
        "runs", nargs="*", type=Path,
        help="a batch folder, a date folder, a single run folder, or two legacy run folders",
    )
    parser.add_argument(
        "--output", default=None,
        help="output PDF path (default: inside each batch folder)",
    )
    args = parser.parse_args()

    if args.runs:
        if len(args.runs) == 1:
            plot_transfer_batches(args.runs[0], output_name=args.output or TRANSFER_PLOT_NAME)
        else:
            plot_transfer(args.runs, output_path=args.output)
    else:
        from race_car.utils.config import get_large_car_params

        car_params = get_small_car_params()
        car_params_sim = get_large_car_params()

        try:
            npz_file = np.load("race_car/jkh_test/small_car_results.npz")
            simX, simU, t = npz_file["simX"], npz_file["simU"], npz_file["t"]

            plot_results_track(simX, car_params, total_time=t[-1])
            plot_results_classic(simX, simU, t, car_params)
            plot_lat_acc(simX, t, car_params)
        except OSError:
            print("Specified log file could not be loaded, skipping result plots.")

        if plt.get_fignums():
            input("Press Enter to continue...")  # keep the plots open
            plt.close('all')

        print("ok")