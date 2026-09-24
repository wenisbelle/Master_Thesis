"""
Slider over the recorded run: the uncertainty map of every drone plus, on the
last panel, the global map of the swarm - the element-wise minimum of all of
them, which is what the cost is computed on.

    python3 logs/plot_states.py                      # logs/recording.pkl
    python3 logs/plot_states.py logs/test_recording.pkl
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.pkl")

##### Okabe-Ito, same order as logs/plot_drones.py so a drone keeps its colour #####
##### across the two figures.                                                  #####
DRONE_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#009E73"]


def view(path=DEFAULT_PATH):
    data = pickle.load(open(path, "rb"))
    meta = data["meta"]
    N, W, H = meta["num_drones"], meta["map_width"], meta["map_height"]
    times, S = meta["times"], data["states"]
    T = len(times)

    ##### Recordings written before the global map was stored are still readable, #####
    ##### the map is rebuilt from the individual ones, which is how it is defined. #####
    global_maps = meta.get("global_maps")

    def global_frame(f):
        if global_maps is not None:
            return global_maps[f]
        return np.min([S[i]["maps"][f] for i in range(N)], axis=0)

    ##### One panel per drone, plus the global one at the end #####
    fig, axes = plt.subplots(1, N + 1, figsize=(5 * (N + 1), 5.5), squeeze=False)
    axes = axes[0]
    drone_axes, global_ax = axes[:N], axes[N]
    fig.subplots_adjust(bottom=0.16)

    ims, selfs, trues = [], [], []
    for i, ax in enumerate(drone_axes):
        ims.append(ax.imshow(S[i]["maps"][0].T, origin="lower", extent=[0, W, 0, H],
                             cmap="gray_r", vmin=0, vmax=1))
        ax.set_title(f"Drone {i}"); ax.set_xlim(0, W); ax.set_ylim(0, H)
        selfs.append(ax.plot([], [], "o", color="tab:blue", ms=9, zorder=3, label="self")[0])
        # true positions of the OTHER drones — green
        trues.append(ax.plot([], [], "+", color="#2ca02c", ms=12, mew=2.5,
                             zorder=4, label="true others")[0])
    drone_axes[0].legend(loc="upper right", fontsize=8)

    ##### The global map carries every drone, each in its own colour: it is the #####
    ##### only panel where they share a frame of reference.                     #####
    global_im = global_ax.imshow(global_frame(0).T, origin="lower", extent=[0, W, 0, H],
                                 cmap="gray_r", vmin=0, vmax=1)
    global_ax.set_title("Global map (min over drones)")
    global_ax.set_xlim(0, W); global_ax.set_ylim(0, H)
    global_dots = [global_ax.plot([], [], "o", color=DRONE_COLORS[i % len(DRONE_COLORS)],
                                  ms=9, zorder=3, label=f"drone {i}")[0]
                   for i in range(N)]
    global_ax.legend(loc="upper right", fontsize=8, ncol=min(N, 3))

    def draw(val):
        f = int(val)
        truth = {i: S[i]["true_pos_cell"][f] for i in range(N)}
        for i in range(N):
            ims[i].set_data(S[i]["maps"][f].T)
            cx, cy = truth[i]; selfs[i].set_data([cx], [cy])
            others = [j for j in range(N) if j != i]
            trues[i].set_data([truth[j][0] for j in others], [truth[j][1] for j in others])
            global_dots[i].set_data([cx], [cy])

        global_map = global_frame(f)
        global_im.set_data(global_map.T)
        global_ax.set_title(f"Global map (min over drones)\n"
                            f"uncertainty {float(global_map.sum()):.0f}")
        fig.suptitle(f"t = {times[f]:.0f}s   ({f + 1}/{T})")
        fig.canvas.draw_idle()

    sl = Slider(fig.add_axes([0.15, 0.05, 0.7, 0.03]), "time", 0, T - 1, valinit=0, valstep=1)
    sl.on_changed(draw); draw(0); plt.show()


if __name__ == "__main__":
    view(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
