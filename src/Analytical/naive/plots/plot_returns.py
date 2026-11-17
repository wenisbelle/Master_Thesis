"""
At which battery the drones choose to go back to the base, and what happens on
the way: they reach it, are sent back to mapping (abort), or die.

    python3 plot_returns.py                        # logs/test_recording.pkl
    python3 plot_returns.py --directory logs_3     # logs_3/test_recording.pkl
    python3 plot_returns.py --output returns       # saves returns.png

The events are recorded by main_test.py when they happen (not on the snapshot
grid), see SimulationRecorder.record_return_event. The naive drone goes back
when its battery drops below RETURN_BATTERY_THRESHOLD, whatever the distance,
and nothing turns it around on the way. Panels:
- A: the battery of every decision, per drone, and the battery it reached the
  base with (the energy the trip took is the line between them).
- B: the same two batteries against the distance to the base at the decision,
  with the battery the threshold leaves on arrival flying straight back. Where
  that line meets the death threshold, 30% is no longer enough.
- One row per drone: its battery over time, with the decisions marked on it.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_drones import (ROOT_DIR, DRONE_COLORS, STATUS_COLORS, status_runs,
                         recording_path, directory_parser)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from returns_metrics import trips, summarize, format_summary

##### Marker of the decision by what came of it. The colour is the drone's. #####
OUTCOME_STYLE = {
    "arrival": dict(marker="v", filled=True, label="reached the base"),
    "abort":   dict(marker="v", filled=False, label="sent back to mapping on the way"),
    "death":   dict(marker="X", filled=True, label="died on the way"),
    "open":    dict(marker="d", filled=False, label="still flying at the end"),
}
BATTERY_COLOR = "0.45"
BAND_ALPHA = 0.15
BAND_STATUSES = ("GOING_TO_BASE", "CHARGING", "DEAD")


def _color(drone_id):
    return DRONE_COLORS[drone_id % len(DRONE_COLORS)]


def _marker_kwargs(style, color, size=55):
    if style["filled"]:
        return dict(marker=style["marker"], s=size, color=color, edgecolors="white", linewidths=0.6)
    return dict(marker=style["marker"], s=size, facecolors="none", edgecolors=color, linewidths=1.3)


def _reference_lines(ax, parameters, horizontal=True):
    """The return threshold and the death threshold, as battery levels."""
    handles = []
    for key, style, text in (("return_battery_threshold", "--", "RETURN_BATTERY_THRESHOLD, back to the base below"),
                             ("dead_battery_threshold", "-.", "dead below")):
        value = parameters.get(key)
        if value is None or not np.isfinite(value):
            continue
        if horizontal:
            ax.axhline(value, color="black", lw=0.9, ls=style)
        handles.append(Line2D([], [], color="black", lw=0.9, ls=style, label=f"{text} ({value:.2f})"))
    return handles


def plot_decision_strip(ax, all_trips, N, parameters, rng):
    for trip in all_trips:
        x = trip["drone"] + rng.uniform(-0.18, 0.18)
        color = _color(trip["drone"])
        if trip["outcome"] == "arrival":
            ax.plot([x, x], [trip["battery"], trip["battery_end"]], color=color, lw=0.8, alpha=0.6)
            ax.scatter([x], [trip["battery_end"]], marker="o", s=18, facecolors="none",
                       edgecolors=color, linewidths=1.0, zorder=3)
        ax.scatter([x], [trip["battery"]], zorder=4,
                   **_marker_kwargs(OUTCOME_STYLE[trip["outcome"]], color))
    handles = _reference_lines(ax, parameters)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"drone {i}" for i in range(N)], fontsize=8)
    ax.set_xlim(-0.6, N - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("battery (fraction of full)")
    ax.set_title("A. Battery when choosing the base (marker) and when reaching it (circle)",
                 loc="left", fontsize=9)
    ax.grid(True, axis="y", color="0.92", lw=0.5)
    return handles


def plot_decision_distance(ax, all_trips, parameters):
    for trip in all_trips:
        distance, color = trip["distance_to_base"], _color(trip["drone"])
        if trip["outcome"] == "arrival":
            ax.plot([distance, distance], [trip["battery"], trip["battery_end"]], color=color, lw=0.8, alpha=0.6)
            ax.scatter([distance], [trip["battery_end"]], marker="o", s=18, facecolors="none",
                       edgecolors=color, linewidths=1.0, zorder=3)
        ax.scatter([distance], [trip["battery"]], zorder=4,
                   **_marker_kwargs(OUTCOME_STYLE[trip["outcome"]], color, size=40))
    _reference_lines(ax, parameters)

    ##### Battery the threshold leaves on arrival, flying straight at the cruise speed.   #####
    ##### Drawn a bit past the farthest trip. The distance where it meets the death     #####
    ##### threshold is usually far beyond the map, so it goes in the label, not the axis. #####
    handles = []
    threshold = parameters.get("return_battery_threshold")
    rate, speed = parameters.get("discharge_rate"), parameters.get("drone_speed")
    dead = parameters.get("dead_battery_threshold")
    distances = [trip["distance_to_base"] for trip in all_trips if np.isfinite(trip["distance_to_base"])]
    if distances and all(value is not None and np.isfinite(value) and value > 0
                         for value in (threshold, rate, speed)):
        d = np.linspace(0, 1.05 * max(distances), 50)
        ax.plot(d, threshold - d / speed * rate, color="black", lw=0.9, ls=":")
        label = "threshold minus the energy to fly straight back"
        if dead is not None and np.isfinite(dead):
            label += f" (dead beyond {(threshold - dead) / rate * speed:.0f} m)"
        handles.append(Line2D([], [], color="black", lw=0.9, ls=":", label=label))

    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel("distance to the base when choosing it (m)")
    ax.set_ylabel("battery (fraction of full)")
    ax.set_title("B. Battery when choosing the base and when reaching it, against the distance",
                 loc="left", fontsize=9)
    ax.grid(True, color="0.92", lw=0.5)
    return handles


def plot_drone_rows(axes, t, S, returns, status_names, parameters):
    seen = set()
    for drone_id, ax in enumerate(axes):
        for value, start, end in status_runs(t, S[drone_id]["status"]):
            name = status_names.get(value)
            if name in BAND_STATUSES:
                ax.axvspan(start, end, color=STATUS_COLORS[name], alpha=BAND_ALPHA, lw=0)
                seen.add(name)
        ax.plot(t, S[drone_id]["battery"], color=BATTERY_COLOR, lw=1.0)
        _reference_lines(ax, parameters)

        color = _color(drone_id)
        for trip in trips({drone_id: returns.get(drone_id, [])}):
            ax.scatter([trip["t"]], [trip["battery"]], zorder=4,
                       **_marker_kwargs(OUTCOME_STYLE[trip["outcome"]], color, size=45))
            if trip["outcome"] == "arrival":
                ax.scatter([trip["t_end"]], [trip["battery_end"]], marker="o", s=20, zorder=4,
                           facecolors="none", edgecolors=color, linewidths=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"drone {drone_id}")
        ax.grid(True, color="0.92", lw=0.5)
        ax.margins(x=0.01)
    return [Patch(facecolor=STATUS_COLORS[name], alpha=BAND_ALPHA,
                  label=name.lower().replace("_", " ")) for name in BAND_STATUSES if name in seen]


def per_drone_table(all_trips, N):
    lines = [f"{'drone':>5} {'decisions':>9} {'arrived':>7} {'aborted':>7} {'died':>4} "
             f"{'battery at decision':>21} {'at arrival':>10}"]
    for drone_id in range(N):
        own = [trip for trip in all_trips if trip["drone"] == drone_id]
        arrived = [trip for trip in own if trip["outcome"] == "arrival"]
        decision = np.array([trip["battery"] for trip in own])
        arrival = np.array([trip["battery_end"] for trip in arrived])
        lines.append(f"{drone_id:>5} {len(own):>9} {len(arrived):>7} "
                     f"{sum(t['outcome'] == 'abort' for t in own):>7} "
                     f"{sum(t['outcome'] == 'death' for t in own):>4} "
                     f"{(f'{decision.mean():.2f} +- {decision.std():.2f}' if len(decision) else '-'):>21} "
                     f"{(f'{arrival.mean():.2f}' if len(arrival) else '-'):>10}")
    return "\n".join(lines)


def plot_returns(path, output=None):
    data = pickle.load(open(path, "rb"))
    meta, S = data["meta"], data["states"]
    t, N = meta["times"], meta["num_drones"]
    print(f"{path}: {N} drones, {len(t)} samples")
    returns = data.get("returns")
    if returns is None:
        print("The recording has no return events, run main_test.py again to record them")
        return
    status_names = {int(value): name for value, name in meta["status_labels"].items()}
    parameters = meta.get("parameters", {})
    all_trips = trips(returns)
    print(format_summary(summarize(returns)))
    print(per_drone_table(all_trips, N))

    ##### Sizes in inches, so the legend and the labels keep their room for any N #####
    title_room, legend_room, gap = 0.8, 1.9, 0.9
    height = 6.5 + 1.1 * N
    plot_area = height - title_room - legend_room
    fig = plt.figure(figsize=(12, height))
    ##### hspace is a fraction of the mean height of the two rows #####
    outer = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.1 * N], hspace=gap / ((plot_area - gap) / 2),
                             top=1 - title_room / height, bottom=legend_room / height,
                             left=0.07, right=0.98)
    top = outer[0].subgridspec(1, 2, wspace=0.18)
    rows = outer[1].subgridspec(N, 1, hspace=0.25)
    rng = np.random.default_rng(0)
    reference_handles = plot_decision_strip(fig.add_subplot(top[0, 0]), all_trips, N, parameters, rng)
    arrival_handles = plot_decision_distance(fig.add_subplot(top[0, 1]), all_trips, parameters)

    axes = [fig.add_subplot(rows[0, 0])]
    axes += [fig.add_subplot(rows[i, 0], sharex=axes[0]) for i in range(1, N)]
    band_handles = plot_drone_rows(axes, t, S, returns, status_names, parameters)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[0].set_title("Battery over time, with the decisions to go to the base", loc="left", fontsize=9)
    axes[-1].set_xlabel("time (s)")

    outcome_handles = [Line2D([], [], ls="", marker=style["marker"], ms=7, color="0.3",
                              markerfacecolor="0.3" if style["filled"] else "none",
                              label=f"decision: {style['label']}")
                       for name, style in OUTCOME_STYLE.items()
                       if any(trip["outcome"] == name for trip in all_trips)]
    outcome_handles.append(Line2D([], [], ls="", marker="o", ms=5, color="0.3", markerfacecolor="none",
                                  label="battery on reaching the base"))
    outcome_handles.append(Line2D([], [], color=BATTERY_COLOR, lw=1.0, label="battery"))
    fig.legend(handles=outcome_handles + reference_handles + arrival_handles + band_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False,
               title="Colour = drone.  The marker is what came of the decision, "
                     "the circle the battery on reaching the base.",
               title_fontsize=8)
    fig.suptitle("When the drones choose to go back to the base", y=1 - 0.25 / height)

    if output:
        fig.savefig(f"{output}.png", dpi=130)
        print(f"Saved {output}.png")
    else:
        plt.show()


if __name__ == "__main__":
    parser = directory_parser("Battery at which the drones choose to go back to the base")
    parser.add_argument("--output", default=None,
                        help="save the figure as <output>.png instead of showing it")
    args = parser.parse_args()
    plot_returns(recording_path(args.directory), output=args.output)
