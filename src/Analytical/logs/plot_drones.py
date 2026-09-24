"""
Plots the per-drone variables that main_test.py reports, over time: battery,
own coverage, map uncertainty (the drone's own and the global one), distance
flown and the status timeline.

    python3 logs/plot_drones.py                      # logs/recording.pkl
    python3 logs/plot_drones.py path/to/recording.pkl

The recording is the one written by Analytical/main_test.py.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.pkl")

##### Okabe-Ito, colourblind-safe. The colour follows the drone id and is never #####
##### recycled, so drone 2 keeps its colour in every panel and in every run.    #####
DRONE_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#009E73"]

##### Reserved for the status strip only, one colour per state #####
STATUS_COLORS = {
    "MAPPING":       "#BDC3C7",
    "GOING_TO_BASE": "#E69F00",
    "CHARGING":      "#009E73",
    "DEAD":          "#D55E00",
}
UNKNOWN_STATUS_COLOR = "#7F8C8D"


def status_runs(times, statuses):
    """Collapses the sampled status into (status value, start, end) bands."""
    runs = []
    for time, status in zip(times, statuses):
        value = int(status) if np.isfinite(status) else -1
        if runs and runs[-1][0] == value:
            runs[-1][2] = time
        else:
            runs.append([value, time, time])

    ##### A band has to reach the next one, otherwise a state that lasted a #####
    ##### single sample would be drawn with zero width.                     #####
    for current, following in zip(runs, runs[1:]):
        current[2] = following[1]
    return runs


def plot_drones(path=DEFAULT_PATH):
    data = pickle.load(open(path, "rb"))
    meta, S = data["meta"], data["states"]
    t, N = meta["times"], meta["num_drones"]
    total_cells = meta["map_width"] * meta["map_height"]
    status_labels = {int(value): name for value, name in meta.get("status_labels", {}).items()}

    ##### One axis per variable, they do not share a scale. The fourth entry is #####
    ##### the swarm level curve drawn on top of the per-drone ones, when the    #####
    ##### variable has one.                                                     #####
    panels = [
        ("battery", lambda s: s["battery"], "battery\n(fraction of full)", None),
        ("coverage", lambda s: 100.0 * s["visited_cells"] / total_cells,
         "cells seen by the drone\n(% of the map)", None),
        ("uncertainty", lambda s: s["uncertainty"], "map uncertainty\n(sum over cells)",
         lambda meta: meta["global_uncertainty"]),
        ("distance", lambda s: s["distance"], "distance flown\n(m)", None),
    ]

    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(10, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(panels) + [0.55]})

    for ax, (_, values_of, ylabel, global_values_of) in zip(axes, panels):
        for drone_id in range(N):
            values = values_of(S[drone_id])
            color = DRONE_COLORS[drone_id % len(DRONE_COLORS)]
            ax.plot(t, values, lw=1.6, color=color, label=f"drone {drone_id}")

            ##### Up to four drones the lines are also labelled at their end, so #####
            ##### identity does not depend on the legend alone.                  #####
            if N <= 4 and len(t):
                ax.annotate(f"{drone_id}", (t[-1], values[-1]), xytext=(4, 0),
                            textcoords="offset points", color=color,
                            fontsize=8, va="center")

        ##### The global map is the minimum over the drones, so its curve always #####
        ##### sits below every other one on this axis.                           #####
        if global_values_of is not None:
            ax.plot(t, global_values_of(meta), color="black", lw=2.0,
                    label="global (min over drones)")
            ##### Headroom, the legend would otherwise sit on top of the curves #####
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, bottom + (top - bottom) * 1.18)
            ax.legend(loc="upper right", fontsize=8, ncol=min(N + 1, 4), frameon=False)

        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.92", lw=0.5)
        ax.margins(x=0.02)

    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 100)
    axes[0].legend(loc="lower left", fontsize=8, ncol=min(N, 4), frameon=False)

    ##### Status strip: one row per drone, the recharges are the bands here #####
    strip = axes[-1]
    seen_statuses = []
    for drone_id in range(N):
        runs = status_runs(t, S[drone_id]["status"])
        strip.broken_barh(
            [(start, end - start) for _, start, end in runs],
            (drone_id - 0.38, 0.76),
            facecolors=[STATUS_COLORS.get(status_labels.get(value), UNKNOWN_STATUS_COLOR)
                        for value, _, _ in runs],
        )
        seen_statuses += [value for value, _, _ in runs]

    strip.set_yticks(range(N))
    strip.set_yticklabels([f"drone {i}" for i in range(N)])
    strip.set_ylim(-0.6, N - 0.4)
    strip.set_xlabel("time (s)")
    strip.grid(True, axis="x", color="0.92", lw=0.5)
    fig.legend(
        handles=[Patch(facecolor=STATUS_COLORS.get(status_labels.get(value), UNKNOWN_STATUS_COLOR),
                       label=status_labels.get(value, f"status {value}"))
                 for value in sorted(set(seen_statuses))],
        loc="lower center", ncol=4, fontsize=8, frameon=False)

    fig.suptitle("Per-drone state over time")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    plt.show()


if __name__ == "__main__":
    plot_drones(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
