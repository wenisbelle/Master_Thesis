"""
Swarm level view of the run: the uncertainty of each drone against the
uncertainty of the global map, and the cells no drone has ever seen.

    python3 logs/plot_metrics.py                      # logs/recording.pkl
    python3 logs/plot_metrics.py logs/test_recording.pkl
"""

import os
import sys
import pickle
import matplotlib.pyplot as plt

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.pkl")

##### Okabe-Ito, same order as logs/plot_drones.py so a drone keeps its colour #####
##### across the figures.                                                      #####
DRONE_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#009E73"]


def plot_metrics(path=DEFAULT_PATH):
    data = pickle.load(open(path, "rb"))
    m, S = data["meta"], data["states"]
    t, gu, uv = m["times"], m["global_uncertainty"], m["unvisited"]
    N = m["num_drones"]
    total = m["map_width"] * m["map_height"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    ##### Each drone only knows its own observations plus whatever it was told #####
    ##### at an encounter, so its curve always sits above the global one. The  #####
    ##### gap between them is what the swarm gains by being a swarm.           #####
    for drone_id in range(N):
        ax1.plot(t, S[drone_id]["uncertainty"], lw=1.1, alpha=0.8,
                 color=DRONE_COLORS[drone_id % len(DRONE_COLORS)],
                 label=f"drone {drone_id}")
    ax1.plot(t, gu, color="black", lw=2.0, label="global (min over drones)")

    ax1.set_ylabel("map uncertainty\n(sum over cells)")
    ax1.grid(True, color="0.92", lw=0.5)
    ##### Headroom for the legend, it would otherwise sit on top of the curves #####
    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom, bottom + (top - bottom) * 1.18)
    ax1.legend(loc="upper right", fontsize=8, ncol=min(N + 1, 4), frameon=False)

    ax2.plot(t, uv, color="tab:red", lw=1.6)
    ax2.set_ylabel("cells never visited\n(by any drone)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(0, total)
    ax2.grid(True, color="0.92", lw=0.5)

    # right-hand axis: same data shown as % coverage, stays synced automatically
    sec = ax2.secondary_yaxis(
        "right",
        functions=(lambda u: 100 * (1 - u / total), lambda c: total * (1 - c / 100)),
    )
    sec.set_ylabel("coverage (%)")

    fig.suptitle("Per-drone and global uncertainty, and coverage over time")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metrics(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
