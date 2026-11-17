"""
Charging of the swarm over time: how many drones are charging at once against
the capacity of the base (Nc), above which the cost adds a congestion penalty.
The naive drone has no charge margin, it goes back below RETURN_BATTERY_THRESHOLD
(plot_returns.py draws the batteries).

    python3 plot_more_metrics.py                        # logs/test_recording.pkl
    python3 plot_more_metrics.py --directory logs_3     # logs_3/test_recording.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

##### Same default file and status colours as the per-drone plot, so a state #####
##### looks the same in both figures.                                         #####
from plot_drones import DEFAULT_PATH, STATUS_COLORS, recording_path, directory_parser


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
    capacity = finite_parameter(meta.get("parameters", {}), "Nc_charging_base")

    ##### Drones charging at once. Above the capacity of the base the cost adds #####
    ##### a congestion penalty for every extra drone.                           #####
    statuses = np.array([S[i]["status"] for i in range(N)])
    charging = (statuses == status_values["CHARGING"]).sum(axis=0)

    fig, ax = plt.subplots(figsize=(10, 2.8))
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
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    ax.grid(True, color="0.92", lw=0.5)
    ax.margins(x=0.02)

    fig.suptitle("Charging over time")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = directory_parser("Charging over time").parse_args()
    plot_more_metrics(recording_path(args.directory))
