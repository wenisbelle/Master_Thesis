"""
Battery management of the swarm over time: how many drones are charging at
once against the capacity of the base, and the effective charge margin of each
drone - the reserve recharge_base_fitness keeps on top of the energy to reach
the base, once the swarm and congestion terms have moved it away from the
tuned CHARGE_MARGIN.

    python3 logs/plot_more_metrics.py                      # logs/test_recording.pkl
    python3 logs/plot_more_metrics.py path/to/recording.pkl

Only recordings written after the margin was added to the recorder carry it,
for older ones only the charging count is drawn.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

##### Same default file, palette and status colours as the per-drone plot, so #####
##### a drone and a state look the same in both figures.                      #####
from plot_drones import DEFAULT_PATH, DRONE_COLORS, STATUS_COLORS, status_runs

BATTERY_COLOR = "0.6"
BAND_ALPHA = 0.15
##### States drawn as background bands. The margin is not used in any of them: #####
##### decisions are only taken while MAPPING.                                  #####
BAND_STATUSES = ("GOING_TO_BASE", "CHARGING", "DEAD")


def finite_parameter(parameters, name):
    value = parameters.get(name)
    return value if value is not None and np.isfinite(value) else None


def plot_more_metrics(path=DEFAULT_PATH):
    data = pickle.load(open(path, "rb"))
    meta, S = data["meta"], data["states"]
    t, N = meta["times"], meta["num_drones"]
    print(f"{path}: {N} drones, {len(t)} samples")
    status_names = {int(value): name for value, name in meta["status_labels"].items()}
    status_values = {name: value for value, name in status_names.items()}
    parameters = meta.get("parameters", {})
    capacity = finite_parameter(parameters, "Nc_charging_base")
    tuned_margin = finite_parameter(parameters, "charge_margin")
    min_margin = finite_parameter(parameters, "min_charge_margin")

    has_margin = "charge_margin" in S[0]
    if not has_margin:
        print("The recording has no charge margin, run Analytical/main_test.py again to record it")

    rows = 1 + (N if has_margin else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.8 + 1.25 * (rows - 1)), sharex=True,
                             squeeze=False, gridspec_kw={"height_ratios": [1.6] + [1] * (rows - 1)})
    axes = axes[:, 0]

    ##### Drones charging at once. Above the capacity of the base the cost adds #####
    ##### a congestion penalty for every extra drone.                           #####
    statuses = np.array([S[i]["status"] for i in range(N)])
    charging = (statuses == status_values["CHARGING"]).sum(axis=0)

    ax = axes[0]
    ax.fill_between(t, charging, step="post", color=STATUS_COLORS["CHARGING"], alpha=0.25, lw=0)
    ax.step(t, charging, where="post", color=STATUS_COLORS["CHARGING"], lw=1.6,
            label="drones charging")
    title = f"Drones charging at once (at most {charging.max()})"
    if capacity is not None:
        ax.axhline(capacity, color="black", lw=1.0, ls="--", label=f"base capacity (Nc = {capacity:g})")
        ##### Samples are taken every few seconds, so this is the sampled estimate #####
        sample_widths = np.diff(t, append=t[-1])
        title += f", {sample_widths[charging > capacity].sum():.0f} s above the capacity"
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_yticks(range(N + 1))
    ##### Headroom, the legend would otherwise sit on top of the curve #####
    ax.set_ylim(0, N + 0.9)
    ax.set_ylabel("drones")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    ax.grid(True, color="0.92", lw=0.5)
    ax.margins(x=0.02)

    if has_margin:
        ##### NaN outside MAPPING, the drone is not deciding anything then #####
        margins = [np.where(S[i]["status"] == status_values["MAPPING"], S[i]["charge_margin"], np.nan)
                   for i in range(N)]
        ##### One scale for every drone. The swarm term can push the margin above #####
        ##### a full battery, which means "go back now".                          #####
        top = max(1.02, 1.05 * np.nanmax(margins)) if np.isfinite(margins).any() else 1.02

        seen_bands = set()
        for drone_id, ax in enumerate(axes[1:]):
            for value, start, end in status_runs(t, S[drone_id]["status"]):
                name = status_names.get(value)
                if name in BAND_STATUSES:
                    ax.axvspan(start, end, color=STATUS_COLORS[name], alpha=BAND_ALPHA, lw=0)
                    seen_bands.add(name)

            ##### Same unit as the margin, both are fractions of a full charge #####
            ax.plot(t, S[drone_id]["battery"], color=BATTERY_COLOR, lw=1.0)
            ax.plot(t, margins[drone_id], color=DRONE_COLORS[drone_id % len(DRONE_COLORS)], lw=1.8)
            if tuned_margin is not None:
                ax.axhline(tuned_margin, color="black", lw=0.9, ls="--")
            if min_margin is not None:
                ax.axhline(min_margin, color="black", lw=0.9, ls=":")

            ax.set_ylim(0, top)
            ax.set_ylabel(f"drone {drone_id}")
            ax.grid(True, color="0.92", lw=0.5)
            ax.margins(x=0.02)

        axes[1].set_title("Effective charge margin and battery (fractions of a full charge)",
                          loc="left", fontsize=9)

        handles = [
            Line2D([], [], color=BATTERY_COLOR, lw=1.0, label="battery"),
            Line2D([], [], color="0.25", lw=1.8,
                   label="effective charge margin (drone colour, only while mapping)"),
        ]
        if tuned_margin is not None:
            handles.append(Line2D([], [], color="black", lw=0.9, ls="--",
                                  label=f"tuned CHARGE_MARGIN ({tuned_margin:g})"))
        if min_margin is not None:
            handles.append(Line2D([], [], color="black", lw=0.9, ls=":",
                                  label=f"MIN_CHARGE_MARGIN ({min_margin:g})"))
        handles += [Patch(facecolor=STATUS_COLORS[name], alpha=BAND_ALPHA,
                          label=name.lower().replace("_", " "))
                    for name in BAND_STATUSES if name in seen_bands]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Charging and charge margin over time")
    fig.tight_layout(rect=(0, 0.05 if has_margin else 0, 1, 1))
    plt.show()


if __name__ == "__main__":
    plot_more_metrics(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
