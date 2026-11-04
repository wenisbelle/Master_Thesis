r"""
Compares runs on two things: the global uncertainty of the swarm over time and
how many drones are charging on each recharge base, against Nc.

    cd src/Analytical/expanding_testing
    python3 plot_charging_bases.py --run swarm/first_test "Swarm, 2 bases"
    python3 plot_charging_bases.py --run swarm/logs_2bases "Swarm" --run egocentric/logs_2bases "Egocentric"
    python3 plot_charging_bases.py --run swarm/logs_2bases "Swarm, 2 bases" \
                                   --run ../swarm/logs "Swarm, conventional" --per-cell
    python3 plot_charging_bases.py --run swarm/first_test "Swarm" --output charging.png

Each --run is a logs directory and the caption of that run in the figure, in
the order the runs are drawn. A relative directory is looked up next to this
file first (swarm/<logs>, egocentric/<logs>), then from the current directory.
An absolute path or the .pkl file itself also works. It does not have to be a
run of expanding_testing: a recording of Analytical/swarm or
Analytical/egocentric (../swarm/logs from here) has a single base and no
charging_base series, so all of its charging drones are counted on base 0.

The figure has one panel of global uncertainty with a line per run and, below
it, one panel per run with the drones charging stacked by base. The top of the
stack is the total over all the bases, the number Nc limits (dashed line). The
periods with more than Nc drones charging are shaded.

Everything comes from test_recording.pkl, sampled every --monitor-interval of
main_test.py (10 s by default). The time above Nc is therefore an estimate
from those samples. The report of main_test.py uses the 1 s samples of the
global map monitor, so the two can differ by a few samples.

The global uncertainty is the sum over the cells of the map. Maps of different
sizes cannot be compared on that sum, so --per-cell divides it by the number of
cells (the mean uncertainty of a cell).
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

##### Relative directories are resolved from here first: the logs of each #####
##### system sit in swarm/<logs> and egocentric/<logs>.                   #####
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
##### Same name main_test.py gives the recording inside the logs directory #####
RECORDING_FILE = "test_recording.pkl"

##### One style per run: colour and dash pattern, so the runs can be told    #####
##### apart without relying on colour (Okabe-Ito, colourblind-safe). The     #####
##### first three hues are kept apart from the base colours below.           #####
RUN_STYLES = [
    {"color": "#0072B2", "ls": "-"},
    {"color": "#D55E00", "ls": "--"},
    {"color": "#CC79A7", "ls": ":"},
    {"color": "#009E73", "ls": "-."},
    {"color": "#E69F00", "ls": (0, (6, 2, 1, 2, 1, 2))},
    {"color": "#56B4E9", "ls": (0, (1, 1))},
]
##### The colour follows the base index, the same in every panel. Base 0 is #####
##### the CHARGING green of Analytical/<system>/plots, so a single base run  #####
##### looks the same as in plot_more_metrics.py.                            #####
BASE_COLORS = ["#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#0072B2", "#D55E00"]
##### Background of the periods with more than Nc drones charging #####
ABOVE_NC_COLOR = "#C0392B"
ABOVE_NC_ALPHA = 0.14
GRID = {"color": "0.92", "lw": 0.5}


def recording_path(directory: str) -> str:
    """The recording of a logs directory, next to this file or from the current directory."""
    directory = os.path.expanduser(directory)
    candidates = [directory] if os.path.isabs(directory) else [os.path.join(ROOT_DIR, directory),
                                                                os.path.abspath(directory)]
    ##### Both are the same when it is run from this directory #####
    candidates = list(dict.fromkeys(os.path.normpath(c) for c in candidates))
    for candidate in candidates:
        path = candidate if candidate.endswith(".pkl") else os.path.join(candidate, RECORDING_FILE)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no {RECORDING_FILE} for {directory!r}, looked in: " + ", ".join(candidates))


def load_run(path: str, caption: str) -> dict:
    """
    Keeps only what is plotted. A recording carries every map of every drone,
    hundreds of MB, and several of them would not fit in memory at once.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    meta, S = data["meta"], data["states"]
    N = meta["num_drones"]
    charging_value = next(int(value) for value, name in meta["status_labels"].items()
                          if name == "CHARGING")
    charging = np.array([S[i]["status"] for i in range(N)]) == charging_value

    base_positions = meta.get("base_positions")
    if "charging_base" in S[0]:
        bases = np.array([S[i]["charging_base"] for i in range(N)])
        number_of_bases = len(base_positions)
    else:
        ##### Recording of Analytical/<system>: one base, every charging drone is on it #####
        bases = np.zeros(charging.shape)
        number_of_bases = 1
        base_positions = None

    per_base = np.array([(charging & (bases == b)).sum(axis=0) for b in range(number_of_bases)])
    total = charging.sum(axis=0)
    if (per_base.sum(axis=0) != total).any():
        print(f"{path}: {int((per_base.sum(axis=0) != total).sum())} samples have a charging drone "
              "on no base, they count in the total but not in the stack")

    capacity = meta.get("parameters", {}).get("Nc_charging_base")
    return {
        "caption": caption,
        "path": path,
        "num_drones": N,
        "cells": meta["map_width"] * meta["map_height"],
        "times": np.asarray(meta["times"], dtype=float),
        "global_uncertainty": np.asarray(meta["global_uncertainty"], dtype=float),
        "per_base": per_base,
        "total": total,
        "capacity": capacity if capacity is not None and np.isfinite(capacity) else None,
        "base_positions": None if base_positions is None else np.asarray(base_positions, dtype=float),
    }


def time_mean(t, values):
    """Mean over the run, trapezoidal."""
    if len(t) < 2 or t[-1] == t[0]:
        return float(np.mean(values))
    return float(np.sum((values[1:] + values[:-1]) / 2 * np.diff(t)) / (t[-1] - t[0]))


def above_capacity(run) -> dict:
    """
    Time with more than Nc drones charging. The samples are drawn as steps, so
    each one holds until the next.
    """
    t, total, capacity = run["times"], run["total"], run["capacity"]
    widths = np.diff(t, append=t[-1])
    duration = max(t[-1] - t[0], 1e-9)
    if capacity is None:
        return {"seconds": np.nan, "share": np.nan, "excess": np.nan, "spans": []}
    above = total > capacity
    ##### Start and end index of every run of samples above Nc #####
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(int), [0]))))
    ends = np.append(t[1:], t[-1])
    return {
        "seconds": float(widths[above].sum()),
        "share": float(100.0 * widths[above].sum() / duration),
        "excess": float((np.maximum(total - capacity, 0) * widths).sum()),
        "spans": [(t[start], ends[stop - 1]) for start, stop in zip(edges[::2], edges[1::2])],
    }


def print_summary(runs, per_cell):
    """The values of the figure, as a table, and where the bases of each run are."""
    unit = "per cell" if per_cell else "sum"
    print()
    print(f"{'run':<28} {'drones':>6} {'bases':>5} {'Nc':>3} {'mean unc':>10} {'final unc':>10} "
          f"{'max chg':>7} {'above Nc':>14} {'drone*s>Nc':>10}  mean charging per base")
    for run in runs:
        scale = run["cells"] if per_cell else 1
        gu = run["global_uncertainty"] / scale
        above = above_capacity(run)
        fmt = (lambda v: f"{v:.4f}") if per_cell else (lambda v: f"{v:.0f}")
        capacity = "-" if run["capacity"] is None else f"{run['capacity']:g}"
        widths = np.diff(run["times"], append=run["times"][-1])
        duration = max(run["times"][-1] - run["times"][0], 1e-9)
        occupancy = "/".join(f"{(counts * widths).sum() / duration:.2f}" for counts in run["per_base"])
        print(f"{run['caption'][:28]:<28} {run['num_drones']:>6} {len(run['per_base']):>5} {capacity:>3} "
              f"{fmt(time_mean(run['times'], gu)):>10} {fmt(gu[-1]):>10} {int(run['total'].max()):>7} "
              f"{above['seconds']:>7.0f} s {above['share']:>4.1f}% {above['excess']:>10.0f}  {occupancy}")
    print(f"(uncertainty: {unit} over the cells; time above Nc from the recording samples)")
    print()
    for run in runs:
        if run["base_positions"] is None:
            print(f"{run['caption']}: one base, position not in the recording")
        else:
            print(f"{run['caption']}: " + ", ".join(f"base {b} ({x:g}, {y:g}) m"
                                                    for b, (x, y) in enumerate(run["base_positions"])))


def plot_charging_bases(runs, per_cell=False, output=None):
    count = len(runs)
    if len({run["cells"] for run in runs}) > 1 and not per_cell:
        print("The maps have different sizes ("
              + ", ".join(f"{run['caption']}: {run['cells']} cells" for run in runs)
              + "), their summed uncertainty is not comparable. Use --per-cell")

    fig, axes = plt.subplots(1 + count, 1, figsize=(10, 3.6 + 1.45 * count), sharex=True,
                             gridspec_kw={"height_ratios": [3] + [1.25] * count})
    ax_uncertainty, charging_axes = axes[0], axes[1:]

    ##### Global uncertainty. The mean over the run goes in the legend: it is the #####
    ##### number the runs are compared on and is hard to read off crossing lines. #####
    for index, run in enumerate(runs):
        t = run["times"]
        gu = run["global_uncertainty"] / (run["cells"] if per_cell else 1)
        mean = time_mean(t, gu)
        ax_uncertainty.plot(t, gu, lw=1.8, **RUN_STYLES[index],
                            label=f"{run['caption']} (mean {mean:.3f})" if per_cell
                            else f"{run['caption']} (mean {mean:.0f})")
    ##### Min over the drones: the cell as the best informed drone knows it #####
    ax_uncertainty.set_ylabel("global uncertainty\n" + ("(mean per cell)" if per_cell else "(sum over cells)"))
    ax_uncertainty.grid(True, **GRID)
    ##### Above the panel, the curves fill the whole height of it #####
    ax_uncertainty.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(count, 3), fontsize=8,
                          frameon=False, handlelength=4, borderaxespad=0.3)
    ax_uncertainty.margins(x=0)

    ##### Drones charging, stacked by base, one panel per run: on a single panel #####
    ##### the steps of small integers would lie on top of each other. Same scale #####
    ##### in every panel, so the heights can be compared.                        #####
    top = int(max(max(run["total"].max(), run["capacity"] or 0) for run in runs))
    number_of_bases = max(len(run["per_base"]) for run in runs)
    seen_above = False
    for ax, run in zip(charging_axes, runs):
        t, per_base, capacity = run["times"], run["per_base"], run["capacity"]
        above = above_capacity(run)
        for start, end in above["spans"]:
            ax.axvspan(start, end, color=ABOVE_NC_COLOR, alpha=ABOVE_NC_ALPHA, lw=0, zorder=0)
            seen_above = True

        ##### No edges and no outline: a sample is a couple of pixels wide, a line #####
        ##### would paint over the short charging spikes. The top of the stack is  #####
        ##### the total over all the bases.                                         #####
        lower = np.zeros(len(t))
        for base, counts in enumerate(per_base):
            upper = lower + counts
            ax.fill_between(t, lower, upper, step="post", facecolor=BASE_COLORS[base], lw=0, zorder=2)
            lower = upper
        if capacity is not None:
            ax.axhline(capacity, color="black", lw=1.0, ls="--", zorder=4)

        ax.set_title(run["caption"], loc="left", fontsize=9, fontweight="bold")
        if capacity is not None:
            ax.set_title(f"Nc = {capacity:g}, more than Nc charging {above['seconds']:.0f} s "
                         f"({above['share']:.1f}% of the run), at most {int(run['total'].max())} at once",
                         loc="right", fontsize=8)
        ax.set_yticks(sorted({0, top} | ({int(capacity)} if capacity is not None else set())))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, top + 0.5)
        ax.set_ylabel("drones\ncharging", fontsize=8)
        ax.grid(True, **GRID)
        ax.margins(x=0)

    ##### Positions are in the printed summary: a recording of Analytical/<system> #####
    ##### does not have them.                                                       #####
    handles = [Patch(facecolor=BASE_COLORS[base], label=f"base {base}") for base in range(number_of_bases)]
    if any(run["capacity"] is not None for run in runs):
        handles.append(Line2D([], [], color="black", lw=1.0, ls="--", label="Nc (over all the bases)"))
    if seen_above:
        handles.append(Patch(facecolor=ABOVE_NC_COLOR, alpha=ABOVE_NC_ALPHA, label="more than Nc charging"))
    columns = min(len(handles), 6)
    fig.legend(handles=handles, loc="lower center", ncol=columns, fontsize=8, frameon=False)

    charging_axes[-1].set_xlabel("time (s)")
    fig.suptitle("Global uncertainty and drones charging on each base")
    legend_rows = -(-len(handles) // columns)
    fig.tight_layout(rect=(0, 0.035 * legend_rows / (1 + 0.25 * count), 1, 1))

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Figure: {os.path.abspath(output)}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Global uncertainty and drones charging on each base, for one or more runs")
    parser.add_argument("--run", nargs=2, action="append", dest="runs", metavar=("DIRECTORY", "CAPTION"),
                        help=f"logs directory of a run and its caption in the figure. A relative directory "
                             f"is looked up in {ROOT_DIR} first, then from the current directory. Repeat it "
                             "for every run to compare")
    parser.add_argument("--per-cell", action="store_true",
                        help="divide the global uncertainty by the number of cells, to compare maps "
                             "of different sizes")
    parser.add_argument("--output", metavar="FILE",
                        help="save the figure to FILE (png, pdf, svg ...) instead of opening a window")
    args = parser.parse_args()

    if not args.runs:
        parser.error("give at least one --run DIRECTORY CAPTION")
    if len(args.runs) > len(RUN_STYLES):
        parser.error(f"at most {len(RUN_STYLES)} runs in one figure, got {len(args.runs)}")

    runs = []
    for directory, caption in args.runs:
        try:
            path = recording_path(directory)
        except FileNotFoundError as error:
            parser.error(str(error))
        print(f"{caption}: loading {path}")
        run = load_run(path, caption)
        if len(run["per_base"]) > len(BASE_COLORS):
            parser.error(f"{path} has {len(run['per_base'])} bases, at most {len(BASE_COLORS)} can be drawn")
        runs.append(run)

    print_summary(runs, args.per_cell)
    plot_charging_bases(runs, per_cell=args.per_cell, output=args.output)


if __name__ == "__main__":
    main()
