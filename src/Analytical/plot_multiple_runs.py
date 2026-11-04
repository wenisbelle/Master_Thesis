"""
Plots the runs multiple_runs.py wrote, for one or more systems, in two figures:

- Global uncertainty: at every time, the mean over the runs (line) and one standard
  deviation around it (band).
- More than Nc drones charging, in three panels that aggregate the runs, so they read
  the same with 10 runs or 200:
  (a) when: the probability of more than Nc drones charging, in windows of --window
      seconds, the mean over the runs of the share of the window above Nc;
  (b) how much: the share of every run above Nc, a dot per run over a box plot, and
      the mean. Two systems seeded alike are compared run by run, run k has the same
      base(s) and wind on both, with a Wilcoxon signed-rank test;
  (c) how many at once: the share of the time with k drones charging, on a log scale,
      everything right of Nc is the congestion, and how deep it goes.
  The intervals are 95% percentile bootstraps of the mean over the runs: they narrow
  as runs are added, the spread between runs (b) does not.

    python3 Analytical/plot_multiple_runs.py --system egocentric
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --range 500 --drones 5   # compared
    python3 Analytical/plot_multiple_runs.py --system expanding_testing/egocentric \
        --logs-directory multiple_runs/3_bases_150
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --save multiple_runs_200_10.pdf
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --window 100

It reads <system>/multiple_runs/<range>/<drones>, the directory multiple_runs.py writes
with the same --range and --drones, or <system>/<logs-directory>. Several systems are
drawn on the same figures, a system without runs there is left out.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import wilcoxon

##### Same layout of the runs, and the same statistics, as the script that writes them #####
from multiple_runs import (SYSTEMS, NUMBER_OF_DRONES, TRANSMISSION_RANGE, CONFIG_FILE, system_dir,
                           default_logs_directory, load_runs, time_grid, aligned, run_statistics)

##### Okabe-Ito, swarm and egocentric keep their colours of comparing_swarm_egocentric.py. #####
##### The lines also differ in dash pattern and the points in marker, so they do not rely #####
##### on colour alone.                                                                    #####
SYSTEM_STYLES = {
    "swarm":                        {"color": "#0072B2", "ls": "-",  "marker": "o"},
    "egocentric":                   {"color": "#D55E00", "ls": "--", "marker": "s"},
    "expanding_testing/swarm":      {"color": "#009E73", "ls": "-.", "marker": "^"},
    "expanding_testing/egocentric": {"color": "#CC79A7", "ls": ":",  "marker": "D"},
}
BAND_ALPHA = 0.25
GRID = {"color": "0.92", "lw": 0.5}

##### Windows of (a), in s. The first recharge wave lasts a few hundred seconds. #####
WINDOW = 50.0
##### Intervals of the means over the runs. Seeded, the figures do not change from call to call. #####
CONFIDENCE = 95
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
JITTER_SEED = 0
OVER_CAPACITY_COLOR = "0.95"


def spread(values):
    """Mean and standard deviation over the runs, a single run has no spread."""
    values = np.asarray(values, dtype=float)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])
    return values.mean(axis=0), std


def bootstrap_mean(values):
    """
    Mean over the runs (first axis) and its percentile bootstrap interval: the runs are
    drawn again with replacement BOOTSTRAP_RESAMPLES times and the mean taken each time.
    """
    values = np.asarray(values, dtype=float)
    runs = len(values)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ##### How many times each run is drawn in each resample: the mean of a resample is a weighted mean #####
    weights = rng.multinomial(runs, np.full(runs, 1.0 / runs), size=BOOTSTRAP_RESAMPLES) / runs
    tail = (100 - CONFIDENCE) / 2
    low, high = np.percentile(weights @ values.reshape(runs, -1), [tail, 100 - tail], axis=0)
    return values.mean(axis=0), low.reshape(values.shape[1:]), high.reshape(values.shape[1:])


def windowed(mask, grid, dt, window):
    """
    Share of every window of window seconds where mask holds, per run (mask has a row per
    run), and the edges of the windows. The last window is shorter when window does not
    divide the run.
    """
    samples = max(1, int(round(window / dt)))
    starts = np.arange(0, mask.shape[1], samples)
    lengths = np.diff(np.r_[starts, mask.shape[1]])
    share = np.add.reduceat(mask.astype(float), starts, axis=1) / lengths
    return share, np.r_[grid[starts], grid[-1] + dt]


def paired_test(loaded):
    """
    p-value of the Wilcoxon signed-rank test of the share above Nc, run k of one system
    against run k of the other, and how many runs were paired. Seeded alike, run k has the
    same base(s) and wind on both. None unless there are two systems seeded alike, with
    some run that differs.
    """
    if len(loaded) != 2:
        return None
    first, second = loaded.values()
    if first["config"]["seed"] != second["config"]["seed"]:
        return None
    shares = [{record["run"]: stats["above_nc"] for record, stats in zip(data["runs"], data["statistics"])}
              for data in (first, second)]
    runs = sorted(set(shares[0]) & set(shares[1]))
    x, y = (np.array([share[run] for run in runs]) for share in shares)
    if not np.any(x != y):
        return None
    return float(wilcoxon(x, y).pvalue), len(runs)


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


def plot_charging(loaded, window):
    ##### When, how much and how many at once: every panel aggregates the runs, so it reads #####
    ##### the same with 10 runs or 200, and the intervals narrow as runs are added.         #####
    fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=(1, 1.25), width_ratios=(1, 1.35))
    plot_charging_in_time(fig.add_subplot(layout[0, :]), loaded, window)
    plot_charging_per_run(fig.add_subplot(layout[1, 0]), loaded)
    plot_charging_at_once(fig.add_subplot(layout[1, 1]), loaded)
    fig.suptitle(f"Drones charging above Nc over the runs, {scenario(loaded)}")
    return fig


def plot_charging_in_time(ax, loaded, window):
    """(a) The probability of more than Nc charging, in windows of window seconds."""
    for system, data in loaded.items():
        style = SYSTEM_STYLES[system]
        above = np.array([series["charging"] > record["nc_charging_base"]
                          for record, series in zip(data["runs"], data["series"])])
        share, edges = windowed(above, data["grid"], data["config"]["sample_interval"], window)
        mean, low, high = bootstrap_mean(100.0 * share)
        ##### The last value is repeated so the step covers the last window too #####
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                        color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax.step(edges, np.r_[mean, mean[-1]], where="post", color=style["color"], ls=style["ls"], lw=1.4,
                label=f"{system}, {len(data['runs'])} runs")
    ax.set_title(f"(a) When: probability of more than Nc drones charging, in {window:g} s windows "
                 f"(mean over the runs, {CONFIDENCE}% bootstrap interval)", loc="left", fontsize=9)
    ax.set_ylabel(r"$P(\mathrm{charging} > N_c)$ (%)")
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, max(data["config"]["duration"] for data in loaded.values()))
    ax.set_ylim(0, min(100, ax.get_ylim()[1]))
    ax.grid(True, **GRID)
    ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=3.5)


def plot_charging_per_run(ax, loaded):
    """(b) The share of each run above Nc: one dot per run over a box plot, and the mean."""
    jitter = np.random.default_rng(JITTER_SEED)
    positions = np.arange(len(loaded))
    for position, (system, data) in zip(positions, loaded.items()):
        style = SYSTEM_STYLES[system]
        shares = np.array([stats["above_nc"] for stats in data["statistics"]])
        line = {"color": style["color"], "lw": 1.0}
        ax.boxplot(shares, positions=[position], widths=0.45, showfliers=False, boxprops=line,
                   whiskerprops=line, capprops=line, medianprops={"color": style["color"], "lw": 2.0})
        ax.scatter(position + jitter.uniform(-0.13, 0.13, len(shares)), shares, marker=style["marker"],
                   s=16, color=style["color"], alpha=0.55, lw=0, zorder=3)
        mean, low, high = bootstrap_mean(shares)
        ax.errorbar(position + 0.36, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black",
                    ms=4, capsize=3, lw=1.0, zorder=4)
        print(f"{system}: more than Nc charging {mean:.2f}% of the run, {CONFIDENCE}% interval "
              f"[{low:.2f}, {high:.2f}] over {len(shares)} runs")

    title = "(b) How much: each run"
    test = paired_test(loaded)
    if test is not None:
        p_value, pairs = test
        compared = ("paired Wilcoxon signed-rank " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
                    + f", {pairs} runs")
        title += "\n" + compared
        print(f"{' vs '.join(loaded)}: {compared}")
    top = max(max(stats["above_nc"] for stats in data["statistics"]) for data in loaded.values())
    ax.set_ylim(0, 1.12 * top if top > 0 else 1.0)
    ax.set_xlim(-0.6, len(loaded) - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([system.replace("/", "/\n") for system in loaded], fontsize=8)
    ax.set_ylabel("time with more than Nc charging\n(% of the run)")
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.45", alpha=0.55, ms=4, label="one run"),
                       Line2D([], [], marker="o", ls="none", color="black", ms=4,
                              label=f"mean, {CONFIDENCE}% interval")],
              loc="upper left", fontsize=7, frameon=False, handletextpad=0.3)


def plot_charging_at_once(ax, loaded):
    """(c) The share of the time with k drones charging, the occupancy of the base(s)."""
    drones = max(data["config"]["drones"] for data in loaded.values())
    counts = np.arange(drones + 1)
    for index, (system, data) in enumerate(loaded.items()):
        style = SYSTEM_STYLES[system]
        occupancy = 100.0 * np.array([np.bincount(series["charging"].astype(int), minlength=drones + 1)
                                      for series in data["series"]]) / len(data["grid"])
        mean, low, high = bootstrap_mean(occupancy)
        ##### Never seen in any run: nothing to draw on a log scale #####
        seen = mean > 0
        ##### Side by side, so the intervals of the systems do not hide each other #####
        offset = (index - (len(loaded) - 1) / 2) * 0.12
        ax.errorbar(counts[seen] + offset, mean[seen], yerr=[(mean - low)[seen], (high - mean)[seen]],
                    marker=style["marker"], ls=style["ls"], color=style["color"], ms=5, lw=1.2, capsize=2,
                    label=system)

    nc = sorted({record["nc_charging_base"] for data in loaded.values() for record in data["runs"]})
    ax.axvspan(nc[0] + 0.5, drones + 0.5, color=OVER_CAPACITY_COLOR, lw=0, zorder=0)
    for value in nc:
        ax.axvline(value + 0.5, color="0.35", ls=":", lw=1.0)
        ax.text(value + 0.58, 0.03, f"more than Nc = {value}", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xticks(counts)
    ax.set_xlim(-0.5, drones + 0.5)
    ax.set_xlabel("drones charging at once (k)")
    ax.set_ylabel("time with k charging\n(% of the run, log scale)")
    ax.set_title("(c) How many at once: occupancy of the base", loc="left", fontsize=9)
    ax.grid(True, axis="y", which="major", **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=3.5)


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
    parser.add_argument("--window", type=float, default=WINDOW,
                        help="width of the windows the probability of more than Nc charging is "
                             "estimated in, in s (default: %(default)s)")
    parser.add_argument("--save",
                        help="also save the figures, as <name>_uncertainty<ext> and <name>_charging<ext>, "
                             "the extension sets the format (pdf, png, ...)")
    args = parser.parse_args()

    loaded = load_systems(args.system, args.logs_directory, args.transmission_range, args.drones)
    figures = {"uncertainty": plot_uncertainty(loaded), "charging": plot_charging(loaded, args.window)}
    if args.save:
        stem, extension = os.path.splitext(args.save)
        for name, fig in figures.items():
            path = f"{stem}_{name}{extension or '.pdf'}"
            fig.savefig(path, bbox_inches="tight")
            print(f"{path}: saved")
    plt.show()
