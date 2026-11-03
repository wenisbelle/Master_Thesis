"""
Plots the runs multiple_runs.py wrote, for one or more systems, in two figures:

- Global uncertainty: at every time, the mean over the runs (line) and one standard
  deviation around it (band).
- More than Nc drones charging: one row per run, marked wherever more than Nc drones
  were charging at once, darker the more drones above Nc, with the share of the run
  that lasted on the right. Below, the share of the runs above Nc at every time.

    python3 Analytical/plot_multiple_runs.py --system egocentric
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --range 500 --drones 5   # compared
    python3 Analytical/plot_multiple_runs.py --system expanding_testing/egocentric \
        --logs-directory multiple_runs/3_bases_150
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --save multiple_runs_200_10.pdf

It reads <system>/multiple_runs/<range>/<drones>, the directory multiple_runs.py writes
with the same --range and --drones, or <system>/<logs-directory>. Several systems are
drawn on the same figures, a system without runs there is left out.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

##### Same layout of the runs, and the same statistics, as the script that writes them #####
from multiple_runs import (SYSTEMS, NUMBER_OF_DRONES, TRANSMISSION_RANGE, CONFIG_FILE, system_dir,
                           default_logs_directory, load_runs, time_grid, aligned, run_statistics)

##### Okabe-Ito, swarm and egocentric keep their colours of comparing_swarm_egocentric.py. #####
##### The mean lines also differ in dash pattern, so they do not rely on colour alone.     #####
SYSTEM_STYLES = {
    "swarm":                        {"color": "#0072B2", "ls": "-"},
    "egocentric":                   {"color": "#D55E00", "ls": "--"},
    "expanding_testing/swarm":      {"color": "#009E73", "ls": "-."},
    "expanding_testing/egocentric": {"color": "#CC79A7", "ls": ":"},
}
BAND_ALPHA = 0.25
GRID = {"color": "0.92", "lw": 0.5}

##### Drones charging above Nc, as tints of the system colour: one, two, three or more #####
EXCESS_LEVELS = (1, 2, 3)
EXCESS_TINTS = (0.35, 0.65, 1.0)
EXCESS_LABELS = ("1 above Nc", "2 above Nc", "3+ above Nc")
TRACK_COLOR = "0.95"
##### Past this many runs only every fifth row is labelled #####
LABELLED_RUNS = 25


def tint(color, amount):
    """color mixed with white, amount 1 is the colour itself. Opaque, so overlaps do not darken."""
    rgb = np.array(to_rgb(color))
    return tuple(1.0 - (1.0 - rgb) * amount)


def spread(values):
    """Mean and standard deviation over the runs, a single run has no spread."""
    values = np.asarray(values, dtype=float)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), std


def stretches(mask, grid, dt):
    """(start, width) of every stretch of grid where mask holds, the way broken_barh takes them."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(grid[start], (end - start) * dt) for start, end in zip(starts, ends)]


def load_systems(systems, logs_directory, transmission_range, number_of_drones):
    loaded = {}
    for system in systems:
        directory = os.path.join(system_dir(system), logs_directory or
                                 default_logs_directory(transmission_range, number_of_drones))
        if not os.path.exists(os.path.join(directory, CONFIG_FILE)):
            print(f"{directory}: no runs, {system} left out")
            continue
        config, runs = load_runs(directory)
        if not runs:
            print(f"{directory}: no runs, {system} left out")
            continue
        print(f"{directory}: {len(runs)} runs")
        grid = time_grid(config)
        loaded[system] = {
            "config": config,
            "grid": grid,
            "runs": runs,
            "series": [aligned(record, grid) for record in runs],
            "statistics": [run_statistics(record, grid) for record in runs],
        }
    if not loaded:
        raise SystemExit("No runs found, run multiple_runs.py first")
    return loaded


def scenario(loaded):
    configs = [data["config"] for data in loaded.values()]

    def values(key):
        return "/".join(f"{value:g}" for value in sorted({config[key] for config in configs}))

    return (f"{values('drones')} drones, {values('transmission_range')} m, {values('duration')} s, "
            f"bases drawn again on every run")


def plot_uncertainty(loaded):
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    for system, data in loaded.items():
        style = SYSTEM_STYLES[system]
        mean, std = spread([series["global_uncertainty"] for series in data["series"]])
        ##### The legend also gives the mean over the whole run, the number the systems #####
        ##### are compared on, with its spread over the runs.                            #####
        run_mean, run_std = spread([stats["mean_uncertainty"] for stats in data["statistics"]])
        ax.fill_between(data["grid"], mean - std, mean + std, color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax.plot(data["grid"], mean, color=style["color"], ls=style["ls"], lw=1.8,
                label=f"{system}, {len(data['runs'])} runs (mean over the run {run_mean:.0f} ± {run_std:.0f})")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("global uncertainty\n(sum over cells)")
    ax.set_title("Global uncertainty (min over drones): mean over the runs, band of ±1 standard deviation",
                 loc="left", fontsize=9)
    ax.set_xlim(0, max(data["config"]["duration"] for data in loaded.values()))
    ax.set_ylim(bottom=0)
    ax.grid(True, **GRID)
    ax.legend(loc="lower right", fontsize=8, frameon=False, handlelength=3.5)
    fig.suptitle(f"Global uncertainty over the runs, {scenario(loaded)}")
    return fig


def plot_charging(loaded):
    ##### One raster per system, a row per run, and the share of runs below them #####
    heights = [max(1.2, 0.22 * len(data["runs"])) for data in loaded.values()] + [1.5]
    fig, axes = plt.subplots(len(loaded) + 1, 1, sharex=True, figsize=(10, 1.4 + sum(heights)),
                             gridspec_kw={"height_ratios": heights}, constrained_layout=True)
    duration = max(data["config"]["duration"] for data in loaded.values())

    for ax, (system, data) in zip(axes, loaded.items()):
        color = SYSTEM_STYLES[system]["color"]
        grid, dt = data["grid"], data["config"]["sample_interval"]
        levels_seen = set()
        for row, (record, series, stats) in enumerate(zip(data["runs"], data["series"], data["statistics"])):
            ##### A light track for the whole run, so a run never above Nc still shows as a row #####
            ax.broken_barh([(0, data["config"]["duration"])], (row - 0.4, 0.8), facecolor=TRACK_COLOR, lw=0)
            excess = np.minimum(series["charging"] - record["nc_charging_base"], EXCESS_LEVELS[-1])
            for level, amount in zip(EXCESS_LEVELS, EXCESS_TINTS):
                spans = stretches(excess == level, grid, dt)
                if spans:
                    levels_seen.add(level)
                    ##### The edge keeps a stretch of a few seconds visible, narrower than a pixel #####
                    ax.broken_barh(spans, (row - 0.4, 0.8), facecolor=tint(color, amount),
                                   edgecolor=tint(color, amount), lw=0.5)
            ax.text(1.005, row, f"{stats['above_nc']:.1f}%", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7)

        runs = len(data["runs"])
        ax.text(1.005, -0.6, "above Nc", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7)
        step = 1 if runs <= LABELLED_RUNS else 5
        ax.set_yticks(range(0, runs, step))
        ax.set_yticklabels([str(record["run"]) for record in data["runs"][::step]], fontsize=7)
        ax.set_ylim(runs - 0.5, -0.5)
        ax.set_ylabel("run")
        ax.grid(True, axis="x", **GRID)
        ax.set_axisbelow(True)

        nc = sorted({record["nc_charging_base"] for record in data["runs"]})
        above_mean, above_std = spread([stats["above_nc"] for stats in data["statistics"]])
        ax.set_title(f"{system}: more than Nc = {'/'.join(str(n) for n in nc)} drones charging "
                     f"{above_mean:.1f} ± {above_std:.1f} % of the run", loc="left", fontsize=9)
        handles = [Patch(facecolor=tint(color, amount), label=label)
                   for level, amount, label in zip(EXCESS_LEVELS, EXCESS_TINTS, EXCESS_LABELS)
                   if level in levels_seen]
        if handles:
            ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles),
                      fontsize=7, frameon=False, borderaxespad=0.2, handlelength=1.5)

    ##### Solid lines: dashes break up into the short steps of the share #####
    ax = axes[-1]
    for system, data in loaded.items():
        color = SYSTEM_STYLES[system]["color"]
        above = np.array([series["charging"] > record["nc_charging_base"]
                          for record, series in zip(data["runs"], data["series"])])
        share = 100.0 * above.mean(axis=0)
        ax.fill_between(data["grid"], share, step="post", color=color, alpha=0.2, lw=0)
        ax.step(data["grid"], share, where="post", color=color, lw=1.2, label=system)
    ax.set_title("Share of the runs with more than Nc drones charging, at every time", loc="left", fontsize=9)
    ax.set_ylabel("runs (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, duration)
    ax.grid(True, **GRID)
    if len(loaded) > 1:
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=len(loaded))

    fig.suptitle(f"Drones charging above Nc over the runs, {scenario(loaded)}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global uncertainty (mean and std deviation) and drones charging above Nc over many runs")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", required=True,
                        help="system(s) to plot, several are drawn on the same figures")
    parser.add_argument("--range", type=float, default=TRANSMISSION_RANGE, dest="transmission_range",
                        help="communication range of the runs, in m (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=NUMBER_OF_DRONES,
                        help="number of drones of the runs (default: %(default)s)")
    parser.add_argument("--logs-directory", "--logs_directory",
                        help="directory of the runs, relative to each system directory "
                             "(default: multiple_runs/<range>/<drones>)")
    parser.add_argument("--save",
                        help="also save the figures, as <name>_uncertainty<ext> and <name>_charging<ext>, "
                             "the extension sets the format (pdf, png, ...)")
    args = parser.parse_args()

    loaded = load_systems(args.system, args.logs_directory, args.transmission_range, args.drones)
    figures = {"uncertainty": plot_uncertainty(loaded), "charging": plot_charging(loaded)}
    if args.save:
        stem, extension = os.path.splitext(args.save)
        for name, fig in figures.items():
            path = f"{stem}_{name}{extension or '.pdf'}"
            fig.savefig(path, bbox_inches="tight")
            print(f"{path}: saved")
    plt.show()
