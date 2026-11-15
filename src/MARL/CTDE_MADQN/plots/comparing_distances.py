"""
Compares the tuned communication distances on one swarm size: the global
uncertainty over the whole run for every TRANSMISSION_RANGE and, below it, how
many drones were charging at each time.

    python3 comparing_distances.py                              # 5 drones, every range in test_logs
    python3 comparing_distances.py --drones 7
    python3 comparing_distances.py --drones 5 --ranges 100 500 2000
    python3 comparing_distances.py --test-logs test_logs_base_400_-200

The recordings are the ones Analytical/all_tests_distance.py writes, in
<test-logs>/<range>/<drones>/test_recording.pkl.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

##### Same root and recording name as the other plots #####
from plot_drones import ROOT_DIR, RECORDING_FILE

DEFAULT_DRONES = 5
DEFAULT_TEST_LOGS = "test_logs"

##### Every range differs in three ways at once - colour, dash pattern and   #####
##### marker - so the lines can be told apart without relying on colour.    #####
##### Okabe-Ito colours, which stay distinct for the common colour blindness #####
##### types, without its yellow, which hardly shows on white.                #####
RANGE_STYLES = [
    {"color": "#000000", "ls": "-",                    "marker": "o"},
    {"color": "#0072B2", "ls": "--",                   "marker": "s"},
    {"color": "#D55E00", "ls": ":",                    "marker": "^"},
    {"color": "#009E73", "ls": "-.",                   "marker": "D"},
    {"color": "#CC79A7", "ls": (0, (6, 2, 1, 2, 1, 2)), "marker": "v"},
    {"color": "#E69F00", "ls": (0, (1, 1)),            "marker": "P"},
    {"color": "#56B4E9", "ls": (0, (8, 3)),            "marker": "X"},
]
##### Markers along each line, sparse so they do not hide the curve, shifted #####
##### per range so two lines do not put theirs on the same samples.          #####
MARKERS_PER_LINE = 12


def range_style(index, samples, count):
    style = dict(RANGE_STYLES[index % len(RANGE_STYLES)])
    step = max(1, samples // MARKERS_PER_LINE)
    style["markevery"] = (index * step // max(count, 1), step)
    return style


def available_ranges(test_logs, number_of_drones):
    """Every range in test_logs with a recording for this swarm size, sorted."""
    ranges = []
    for name in os.listdir(test_logs):
        try:
            transmission_range = float(name)
        except ValueError:
            continue
        if os.path.exists(os.path.join(test_logs, name, str(number_of_drones), RECORDING_FILE)):
            ranges.append(transmission_range)
    return sorted(ranges)


def load_run(path):
    """
    Only the swarm level series are kept: a recording carries every map of
    every drone, hundreds of MB, and all of them would not fit at once.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    meta, S = data["meta"], data["states"]
    charging_value = next(int(value) for value, name in meta["status_labels"].items()
                          if name == "CHARGING")
    statuses = np.array([S[i]["status"] for i in range(meta["num_drones"])])
    capacity = meta.get("parameters", {}).get("Nc_charging_base")
    return {
        "times": np.asarray(meta["times"]),
        "global_uncertainty": np.asarray(meta["global_uncertainty"]),
        "charging": (statuses == charging_value).sum(axis=0),
        "capacity": capacity if capacity is not None and np.isfinite(capacity) else None,
    }


def comparing_distances(number_of_drones=DEFAULT_DRONES, ranges=None, test_logs=DEFAULT_TEST_LOGS):
    test_logs = os.path.join(ROOT_DIR, test_logs)
    if ranges is None:
        ranges = available_ranges(test_logs, number_of_drones)
    runs = {}
    for transmission_range in sorted(ranges):
        path = os.path.join(test_logs, f"{transmission_range:g}", str(number_of_drones), RECORDING_FILE)
        if not os.path.exists(path):
            print(f"{path}: not found, {transmission_range:g} m left out")
            continue
        print(f"{path}: loading")
        runs[transmission_range] = load_run(path)
    if not runs:
        raise SystemExit(f"No recording with {number_of_drones} drones in {test_logs}")

    ##### Below the uncertainty, one charging panel per range. On a single panel #####
    ##### the step lines of small integers would lie on top of each other.     #####
    fig, axes = plt.subplots(1 + len(runs), 1, figsize=(10, 4.2 + 0.75 * len(runs)), sharex=True,
                             gridspec_kw={"height_ratios": [4] + [1] * len(runs)},
                             constrained_layout=True)
    ax1, charging_axes = axes[0], axes[1:]

    ##### The mean over the run goes in the legend, it is the number the curves #####
    ##### are compared on and is hard to read off them when they cross.          #####
    for index, (transmission_range, run) in enumerate(runs.items()):
        t, gu = run["times"], run["global_uncertainty"]
        mean = np.trapz(gu, t) / (t[-1] - t[0]) if len(t) > 1 else gu.mean()
        ax1.plot(t, gu, lw=1.8, ms=6, **range_style(index, len(t), len(runs)),
                 label=f"{transmission_range:g} m (mean {mean:.0f})")
    ax1.set_ylabel("global uncertainty\n(sum over cells)")
    ax1.set_title("Global uncertainty (min over drones)", loc="left", fontsize=9)
    ax1.grid(True, color="0.92", lw=0.5)
    ##### One legend for both panels, a range has the same line in each. The #####
    ##### curves start low and rise, the lower right corner stays empty.     #####
    ax1.legend(title="communication range", loc="lower right", fontsize=8, title_fontsize=8,
               ncol=2, frameon=False, handlelength=4)

    ##### Drones charging at once, as in plot_more_metrics. Above the capacity #####
    ##### of the base the cost adds a congestion penalty for every extra drone. #####
    capacities = {run["capacity"] for run in runs.values()}
    capacity = capacities.pop() if len(capacities) == 1 and None not in capacities else None
    top = max(run["charging"].max() for run in runs.values())
    for index, (ax, (transmission_range, run)) in enumerate(zip(charging_axes, runs.items())):
        t, charging = run["times"], run["charging"]
        ##### One range per panel, named inside it, so a solid line in its colour #####
        ##### is enough here: dashes break up the short charging steps.           #####
        color = RANGE_STYLES[index % len(RANGE_STYLES)]["color"]
        ax.fill_between(t, charging, step="post", color=color, alpha=0.2, lw=0)
        ax.step(t, charging, where="post", color=color, lw=1.2)
        if capacity is not None:
            ax.axhline(capacity, color="black", lw=0.9, ls="--")
        ax.text(0.005, 0.92, f"{transmission_range:g} m", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_yticks(sorted({0, top} | ({capacity} if capacity is not None else set())))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, top + 0.4)
        ax.grid(True, color="0.92", lw=0.5)
    fig.supylabel("drones charging", x=0, y=0.27, fontsize=10)

    title = "Drones charging at once, per communication range"
    if capacity is not None:
        title += f" (dashed: base capacity Nc = {capacity:g})"
    charging_axes[0].set_title(title, loc="left", fontsize=9)
    charging_axes[-1].set_xlabel("time (s)")
    ax1.margins(x=0)

    fig.suptitle(f"Communication ranges compared, {number_of_drones} drones")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global uncertainty and drones charging over time, for every communication range")
    parser.add_argument("--drones", type=int, default=DEFAULT_DRONES,
                        help="swarm size to compare (default: %(default)s)")
    parser.add_argument("--ranges", type=float, nargs="+",
                        help="communication ranges to compare, in m (default: every range "
                             "in --test-logs with a run of that swarm size)")
    parser.add_argument("--test-logs", default=DEFAULT_TEST_LOGS,
                        help="directory written by all_tests_distance.py, relative to "
                             f"{ROOT_DIR} (default: %(default)s)")
    args = parser.parse_args()
    comparing_distances(args.drones, args.ranges, args.test_logs)
