"""
How wrong the drones are about each other's battery: what each drone believes
(its swarm_battery_status, projected by estimate_drone_battery) against the
true state, which only the recorder sees.

    python3 plot_estimation.py                        # logs/test_recording.pkl
    python3 plot_estimation.py --directory logs_3     # logs_3/test_recording.pkl
    python3 plot_estimation.py --drone 2              # the beliefs of drone 2 in detail
    python3 plot_estimation.py --output est           # saves est_overview.png, est_drone_0.png

Two figures:
- Overview: estimated against true battery of the others, the error against the
  age of the information, the error of the number of drones charging and of the
  effective charge margin, and the error of both swarm terms over time.
- One drone (--drone): the true battery of every other drone next to what that
  drone believes it is, and when it believes it is charging.

Every error is measured while the observer can decide (MAPPING or GOING_TO_BASE),
see estimation_metrics.py, which also prints the numbers.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_drones import (ROOT_DIR, DRONE_COLORS, STATUS_COLORS, status_runs,
                         recording_path, directory_parser)

##### estimation_metrics.py sits next to main_test.py, one level up #####
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from estimation_metrics import (pair_samples, swarm_term_samples, summarize, format_summary,
                                has_beliefs, MARGIN_TOLERANCE, DECIDING)

##### Marker and colour of the true state of the target drone. MAPPING is darker #####
##### than its strip colour, the strip grey does not show as a marker.           #####
TARGET_STYLE = {
    "MAPPING":       dict(color="0.35", marker="o"),
    "GOING_TO_BASE": dict(color=STATUS_COLORS["GOING_TO_BASE"], marker="s"),
    "CHARGING":      dict(color=STATUS_COLORS["CHARGING"], marker="^"),
}
##### Okabe-Ito blue / vermillion around a grey zero: estimate too low / too high #####
UNDER_COLOR, OVER_COLOR, ZERO_COLOR = "#0072B2", "#D55E00", "#E8E8E8"
NOT_DECIDING_COLOR = "white"
MAX_POINTS = 4000
BAND_ALPHA = 0.15


def _grid(ax):
    ax.grid(True, color="0.92", lw=0.5)


def _subsample(n, limit, rng):
    return np.arange(n) if n <= limit else np.sort(rng.choice(n, limit, replace=False))


def _status_maps(meta):
    names = {int(value): name for value, name in meta["status_labels"].items()}
    return names, {name: value for value, name in names.items()}


def plot_estimated_vs_true(ax, pairs, status_values, rng):
    for name, style in TARGET_STYLE.items():
        mask = pairs["true_status"] == status_values.get(name, np.nan)
        index = np.nonzero(mask)[0]
        index = index[_subsample(len(index), MAX_POINTS, rng)]
        ax.scatter(pairs["true"][index], pairs["estimate"][index], s=9, alpha=0.35, lw=0,
                   color=style["color"], marker=style["marker"],
                   label=f"target {name.lower().replace('_', ' ')} (n={mask.sum()})")
    ax.plot([0, 1], [0, 1], color="black", lw=0.9, ls="--", label="estimate = truth")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("true battery of the other drone")
    ax.set_ylabel("battery the observer believes")
    ax.set_title("A. Battery of the others: belief against truth", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False, markerscale=1.6)
    _grid(ax)


def plot_error_vs_age(ax, pairs):
    age, error = pairs["age"].astype(float), pairs["error"]
    finite = np.isfinite(age) & np.isfinite(error)
    age, error = age[finite], error[finite]
    if not len(age):
        return
    edges = np.linspace(0, np.percentile(age, 99), 16)
    centres, rmse, bias, p10, p90 = [], [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (age >= low) & (age < high)
        if in_bin.sum() < 10:
            continue
        centres.append((low + high) / 2)
        rmse.append(np.sqrt(np.mean(error[in_bin] ** 2)))
        bias.append(np.mean(error[in_bin]))
        p10.append(np.percentile(error[in_bin], 10))
        p90.append(np.percentile(error[in_bin], 90))

    ax.fill_between(centres, p10, p90, color="0.6", alpha=0.25, lw=0, label="10-90% of the errors")
    ax.plot(centres, rmse, color="black", lw=1.8, marker="o", ms=4, label="RMSE")
    ax.plot(centres, bias, color="0.4", lw=1.6, ls="--", marker="s", ms=4,
            label="bias (mean of estimate - truth)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("age of the information (s since the row was written)")
    ax.set_ylabel("estimate - truth (fraction of a battery)")
    ax.set_title("B. Error against the age of the information", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    _grid(ax)


def plot_count_error(ax, terms):
    error = terms["drones_charging_estimate"] - terms["drones_charging_true"]
    error = error[np.isfinite(error)].astype(int)
    if not len(error):
        return
    values = np.arange(error.min(), error.max() + 1)
    shares = np.array([np.mean(error == v) for v in values])
    colors = [UNDER_COLOR if v < 0 else OVER_COLOR if v > 0 else "0.45" for v in values]
    ax.bar(values, 100 * shares, width=0.8, color=colors, edgecolor="white", lw=2)
    for v, share in zip(values, shares):
        if share > 0:
            ax.annotate(f"{100 * share:.0f}%", (v, 100 * share), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:+d}" if v else "0\nexact" for v in values])
    ax.set_ylim(0, 115)
    ax.set_xlabel("drones charging: estimate - truth  (< 0 misses a charging drone)")
    ax.set_ylabel("% of the decision samples")
    ax.set_title("C. Number of drones charging, as the congestion term sees it", loc="left", fontsize=9)
    ax.grid(True, axis="y", color="0.92", lw=0.5)


def plot_margin(ax, terms, parameters, rng):
    estimate, true = terms["charge_margin_estimate"], terms["charge_margin_true"]
    finite = np.isfinite(estimate) & np.isfinite(true)
    estimate, true = estimate[finite], true[finite]
    if not len(estimate):
        ax.text(0.5, 0.5, "no charge margin in the recording", ha="center", transform=ax.transAxes)
        return
    index = _subsample(len(estimate), MAX_POINTS, rng)
    top = max(1.0, 1.05 * max(estimate.max(), true.max()))
    ax.fill_between([0, top], [-MARGIN_TOLERANCE, top - MARGIN_TOLERANCE],
                    [MARGIN_TOLERANCE, top + MARGIN_TOLERANCE], color="0.6", alpha=0.2, lw=0,
                    label=f"within +-{MARGIN_TOLERANCE}")
    ax.scatter(true[index], estimate[index], s=9, alpha=0.35, lw=0, color=DRONE_COLORS[0])
    ax.plot([0, top], [0, top], color="black", lw=0.9, ls="--", label="estimate = truth")
    min_margin = parameters.get("min_charge_margin")
    if min_margin is not None and np.isfinite(min_margin):
        ax.axhline(min_margin, color="black", lw=0.8, ls=":", label=f"MIN_CHARGE_MARGIN ({min_margin:g})")
        ax.axvline(min_margin, color="black", lw=0.8, ls=":")
    changed = np.mean(np.abs(estimate - true) > MARGIN_TOLERANCE)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("margin with the true swarm terms")
    ax.set_ylabel("margin with the estimated ones")
    ax.set_title(f"D. Effective charge margin\n(off by > {MARGIN_TOLERANCE} "
                 f"in {100 * changed:.0f}% of the samples)", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    _grid(ax)


def _deciding_matrix(meta, S, status_values, values_of):
    """(N, T) of values_of(drone), NaN where the drone is not deciding."""
    deciding = [status_values[name] for name in DECIDING if name in status_values]
    return np.array([np.where(np.isin(S[i]["status"], deciding), values_of(S[i]), np.nan)
                     for i in range(meta["num_drones"])], dtype=float)


def plot_error_heatmap(ax, fig, t, matrix, cmap, norm, title, colorbar_label, ticks=None):
    N = matrix.shape[0]
    ax.set_facecolor(NOT_DECIDING_COLOR)
    image = ax.pcolormesh(np.append(t, t[-1]), np.arange(N + 1) - 0.5, np.ma.masked_invalid(matrix),
                          cmap=cmap, norm=norm, shading="flat")
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"drone {i}" for i in range(N)], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.03, ticks=ticks)
    colorbar.set_label(colorbar_label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def plot_overview(data, rng):
    meta, S = data["meta"], data["states"]
    t, N = meta["times"], meta["num_drones"]
    status_names, status_values = _status_maps(meta)
    parameters = meta.get("parameters", {})
    pairs = pair_samples(S, status_values)
    terms = swarm_term_samples(S, status_values)

    fig = plt.figure(figsize=(12, 13))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.25, 1.0, 0.55 + 0.1 * N, 0.55 + 0.1 * N])
    plot_estimated_vs_true(fig.add_subplot(grid[0, 0]), pairs, status_values, rng)
    plot_error_vs_age(fig.add_subplot(grid[0, 1]), pairs)
    plot_count_error(fig.add_subplot(grid[1, 0]), terms)
    plot_margin(fig.add_subplot(grid[1, 1]), terms, parameters, rng)

    ##### The two swarm terms over time, per drone. White is a drone that is not #####
    ##### deciding (charging or dead), its estimate is not used then.            #####
    mean_error = _deciding_matrix(meta, S, status_values,
                                  lambda s: s["mean_battery_estimate"] - s["mean_battery_true"])
    limit = max(0.05, float(np.nanpercentile(np.abs(mean_error), 99))) if np.isfinite(mean_error).any() else 0.1
    ax_mean = fig.add_subplot(grid[2, :])
    plot_error_heatmap(ax_mean, fig, t, mean_error, plt.get_cmap("PuOr_r"),
                       plt.Normalize(-limit, limit),
                       "E. Swarm mean battery: estimate - truth (purple: believes the others have less)",
                       "estimate - truth")

    count_error = _deciding_matrix(meta, S, status_values,
                                   lambda s: s["drones_charging_estimate"] - s["drones_charging_true"])
    largest = int(np.nanmax(np.abs(count_error))) if np.isfinite(count_error).any() else 1
    largest = max(1, largest)
    levels = np.arange(-largest, largest + 1)
    colors = [UNDER_COLOR if v < 0 else OVER_COLOR if v > 0 else ZERO_COLOR for v in levels]
    ##### Darker the further from 0, so the size of the miss does not rely on hue #####
    alphas = [0.35 + 0.65 * abs(v) / largest if v else 1.0 for v in levels]
    cmap = ListedColormap([(*plt.matplotlib.colors.to_rgb(c), a) for c, a in zip(colors, alphas)])
    ax_count = fig.add_subplot(grid[3, :], sharex=ax_mean)
    plot_error_heatmap(ax_count, fig, t, count_error, cmap,
                       BoundaryNorm(np.append(levels, largest + 1) - 0.5, len(levels)),
                       "F. Drones charging: estimate - truth (blue: misses a charging drone, "
                       "vermillion: counts one that is not)", "estimate - truth", ticks=levels)
    ax_count.set_xlabel("time (s)")
    ax_mean.tick_params(labelbottom=False)

    fig.suptitle(f"Battery estimation against the truth ({N} drones, white: drone not deciding)")
    fig.tight_layout()
    return fig


def plot_drone_beliefs(data, observer):
    """The true battery of every other drone and what observer believes it is."""
    meta, S = data["meta"], data["states"]
    t, N = meta["times"], meta["num_drones"]
    status_names, status_values = _status_maps(meta)
    others = [j for j in range(N) if j != observer]
    belief = S[observer]["belief_battery"]
    believed_charging = S[observer]["belief_charging"] > 0.5
    observer_color = DRONE_COLORS[observer % len(DRONE_COLORS)]

    fig, axes = plt.subplots(len(others), 1, figsize=(11, 1.0 + 1.35 * len(others)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    seen = set()
    for ax, j in zip(axes, others):
        ##### True state of the target as background bands #####
        for value, start, end in status_runs(t, S[j]["status"]):
            name = status_names.get(value)
            if name in ("GOING_TO_BASE", "WAITING", "CHARGING", "DEAD"):
                ax.axvspan(start, end, color=STATUS_COLORS[name], alpha=BAND_ALPHA, lw=0)
                seen.add(name)
        ##### Where the observer believes the target charges: a hatched strip at the bottom #####
        for value, start, end in status_runs(t, believed_charging[:, j].astype(float)):
            if value == 1:
                ax.axvspan(start, end, ymin=0, ymax=0.1, facecolor="none",
                           edgecolor=STATUS_COLORS["CHARGING"], hatch="////", lw=0)
                seen.add("believed")
        ##### The observer's own non-deciding periods, its beliefs are not used then #####
        for value, start, end in status_runs(t, S[observer]["status"]):
            if status_names.get(value) in ("CHARGING", "DEAD"):
                ax.axvspan(start, end, ymin=0.92, ymax=1.0, color="0.3", lw=0)
                seen.add("observer_idle")

        ax.plot(t, S[j]["battery"], color="black", lw=1.3)
        ax.plot(t, belief[:, j], color=observer_color, lw=1.6, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"drone {j}")
        _grid(ax)
        ax.margins(x=0.01)

    handles = [Line2D([], [], color="black", lw=1.3, label="true battery of the drone"),
               Line2D([], [], color=observer_color, lw=1.6, ls="--",
                      label=f"what drone {observer} believes")]
    handles += [Patch(facecolor=STATUS_COLORS[name], alpha=BAND_ALPHA,
                      label=f"really {name.lower().replace('_', ' ')}")
                for name in ("GOING_TO_BASE", "WAITING", "CHARGING", "DEAD") if name in seen]
    if "believed" in seen:
        handles.append(Patch(facecolor="none", edgecolor=STATUS_COLORS["CHARGING"], hatch="////",
                             label=f"drone {observer} believes it charges"))
    if "observer_idle" in seen:
        handles.append(Patch(facecolor="0.3", label=f"drone {observer} charging or dead (not deciding)"))
    axes[-1].set_xlabel("time (s)")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Battery of the other drones: truth and the belief of drone {observer}")
    fig.tight_layout(rect=(0, 0.04 + 0.01 * (len(handles) > 3), 1, 1))
    return fig


def plot_estimation(path, observer=0, output=None):
    data = pickle.load(open(path, "rb"))
    meta, S = data["meta"], data["states"]
    print(f"{path}: {meta['num_drones']} drones, {len(meta['times'])} samples")
    if not has_beliefs(S):
        print("The recording has no battery estimate: it is egocentric, or it was recorded before "
              "the beliefs were added. Run main_test.py again to record them.")
        return
    _, status_values = _status_maps(meta)
    print(format_summary(summarize(S, data.get("returns", {}), status_values)))

    rng = np.random.default_rng(0)
    figures = {"overview": plot_overview(data, rng),
               f"drone_{observer}": plot_drone_beliefs(data, observer)}
    if output:
        for name, fig in figures.items():
            fig.savefig(f"{output}_{name}.png", dpi=130)
            print(f"Saved {output}_{name}.png")
    else:
        plt.show()


if __name__ == "__main__":
    parser = directory_parser("Battery estimation of the drones against the truth")
    parser.add_argument("--drone", type=int, default=0,
                        help="drone whose beliefs are drawn in detail (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="save the figures as <output>_overview.png and <output>_drone_<k>.png "
                             "instead of showing them")
    args = parser.parse_args()
    plot_estimation(recording_path(args.directory), observer=args.drone, output=args.output)
