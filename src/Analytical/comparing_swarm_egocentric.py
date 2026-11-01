"""
Compares the swarm and the egocentric systems on one communication range and one
swarm size: the global uncertainty over the whole run of each system and, below
it, how many drones of each were charging at each time.

    python3 Analytical/comparing_swarm_egocentric.py --range 200 --drones 5
    python3 Analytical/comparing_swarm_egocentric.py --range 200 --drones 5 --test-logs test_logs_base_400_-200

It can be started from any directory. The recordings are the ones
all_tests_distance.py writes, in <system>/<test-logs>/<range>/<drones>/test_recording.pkl,
run it with --system swarm and with --system egocentric first.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

##### Same layout of the test logs as the runs that write them #####
from all_tests_distance import SYSTEMS, RECORDING_FILE, system_dir

##### The two systems differ in three ways at once - colour, dash pattern and #####
##### marker - so the lines can be told apart without relying on colour.     #####
##### Okabe-Ito colours, as in swarm/plots/comparing_distances.py.            #####
SYSTEM_STYLES = {
    "swarm":      {"color": "#0072B2", "ls": "-",  "marker": "o"},
    "egocentric": {"color": "#D55E00", "ls": "--", "marker": "s"},
}
##### Markers along each line, sparse so they do not hide the curve, shifted #####
##### per system so the two lines do not put theirs on the same samples.     #####
MARKERS_PER_LINE = 12


def system_style(system, samples):
    index = SYSTEMS.index(system)
    style = dict(SYSTEM_STYLES[system])
    step = max(1, samples // MARKERS_PER_LINE)
    style["markevery"] = (index * step // len(SYSTEMS), step)
    return style


def recording(system, transmission_range, number_of_drones, test_logs):
    return os.path.join(system_dir(system), test_logs, f"{transmission_range:g}",
                        str(number_of_drones), RECORDING_FILE)


def load_run(path):
    """
    Only the swarm level series are kept: a recording carries every map of
    every drone, hundreds of MB, and both of them would hardly fit at once.
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


def comparing_swarm_egocentric(transmission_range, number_of_drones, test_logs="test_logs"):
    runs = {}
    for system in SYSTEMS:
        path = recording(system, transmission_range, number_of_drones, test_logs)
        if not os.path.exists(path):
            print(f"{path}: not found, {system} left out")
            continue
        print(f"{path}: loading")
        runs[system] = load_run(path)
    if not runs:
        raise SystemExit(f"No recording of {transmission_range:g} m with {number_of_drones} drones "
                         f"in any {test_logs}, run all_tests_distance.py first")

    ##### Below the uncertainty, one charging panel per system. On a single  #####
    ##### panel the step lines of small integers would lie on top of each other. #####
    fig, axes = plt.subplots(1 + len(runs), 1, figsize=(10, 4.2 + 0.75 * len(runs)), sharex=True,
                             gridspec_kw={"height_ratios": [4] + [1] * len(runs)},
                             constrained_layout=True)
    ax1, charging_axes = axes[0], axes[1:]

    ##### The mean over the run goes in the legend, it is the number the curves #####
    ##### are compared on and is hard to read off them when they cross.          #####
    for system, run in runs.items():
        t, gu = run["times"], run["global_uncertainty"]
        mean = np.trapz(gu, t) / (t[-1] - t[0]) if len(t) > 1 else gu.mean()
        ax1.plot(t, gu, lw=1.8, ms=6, **system_style(system, len(t)),
                 label=f"{system} (mean {mean:.0f})")
    ax1.set_ylabel("global uncertainty\n(sum over cells)")
    ax1.set_title("Global uncertainty (min over drones)", loc="left", fontsize=9)
    ax1.grid(True, color="0.92", lw=0.5)
    ##### Both curves start from the same low uncertainty, the upper left corner stays empty #####
    ax1.legend(title="system", loc="upper left", fontsize=8, title_fontsize=8,
               frameon=False, handlelength=4)

    ##### Drones charging at once, as in plot_more_metrics. Above the capacity #####
    ##### of the base the cost adds a congestion penalty for every extra drone. #####
    capacities = {run["capacity"] for run in runs.values()}
    capacity = capacities.pop() if len(capacities) == 1 and None not in capacities else None
    top = max(run["charging"].max() for run in runs.values())
    for ax, (system, run) in zip(charging_axes, runs.items()):
        t, charging = run["times"], run["charging"]
        ##### One system per panel, named inside it, so a solid line in its colour #####
        ##### is enough here: dashes break up the short charging steps.            #####
        color = SYSTEM_STYLES[system]["color"]
        ax.fill_between(t, charging, step="post", color=color, alpha=0.2, lw=0)
        ax.step(t, charging, where="post", color=color, lw=1.2)
        if capacity is not None:
            ax.axhline(capacity, color="black", lw=0.9, ls="--")
        ax.text(0.005, 0.92, system, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_yticks(sorted({0, top} | ({capacity} if capacity is not None else set())))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, top + 0.4)
        ax.grid(True, color="0.92", lw=0.5)
    fig.supylabel("drones charging", x=0, y=0.27, fontsize=10)

    title = "Drones charging at once, per system"
    if capacity is not None:
        title += f" (dashed: base capacity Nc = {capacity:g})"
    charging_axes[0].set_title(title, loc="left", fontsize=9)
    charging_axes[-1].set_xlabel("time (s)")
    ax1.margins(x=0)

    fig.suptitle(f"Swarm and egocentric compared, {transmission_range:g} m, {number_of_drones} drones")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global uncertainty and drones charging over time, swarm against egocentric")
    parser.add_argument("--range", type=float, required=True, dest="transmission_range",
                        help="communication range to compare, in m")
    parser.add_argument("--drones", type=int, required=True,
                        help="swarm size to compare")
    parser.add_argument("--test-logs", default="test_logs",
                        help="directory written by all_tests_distance.py, relative to each "
                             "system directory (default: %(default)s)")
    args = parser.parse_args()
    comparing_swarm_egocentric(args.transmission_range, args.drones, args.test_logs)
