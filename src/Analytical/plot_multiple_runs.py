"""
Plots the runs multiple_runs.py wrote, for one or more systems, in two figures:

- Global uncertainty: at every time, the mean over the runs (line) and one standard
  deviation around it (band).
- Global uncertainty relative to a reference system (--reference, naive by default), in %,
  when there are several systems: the absolute figure hides differences of a few %. Seeded
  alike, run k of each system is compared with run k of the reference (same base(s) and
  wind), which removes the spread the base position adds to both. (a) in time, in windows
  of --uncertainty-window seconds; (b) the mean of each run from --steady-state on, a dot
  per run, with a paired Wilcoxon signed-rank test.
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
- Time between two visits of a cell, by the distance from the cell to its nearest base,
  in bins of --distance-bin metres (see multiple_runs.RevisitMonitor for what a visit is):
  the mean (line) and one standard deviation around it (band), over every revisit of the
  cells at that distance in every run. The bases move from run to run, so a distance is
  only drawn where at least half of the runs revisited a cell there. Cells seen at most
  once have no time between two visits and are left out, their share is printed. Runs
  written before the revisits were recorded, and the MARL ones, do not have them.

    python3 Analytical/plot_multiple_runs.py --system egocentric
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --range 500 --drones 5   # compared
    python3 Analytical/plot_multiple_runs.py --system expanding_testing/egocentric \
        --logs-directory multiple_runs/3_bases_150
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --save multiple_runs_200_10.pdf
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --window 100
    python3 Analytical/plot_multiple_runs.py --system swarm egocentric --distance-bin 200
    python3 Analytical/plot_multiple_runs.py --system swarm MARL/CTDE_MADQN --range 200 --drones 5   # RL runs
                                             # of src/MARL/CTDE_MADQN/multiple_runs.py

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
from multiple_runs import (SYSTEMS, MARL_SYSTEMS, NUMBER_OF_DRONES, TRANSMISSION_RANGE, CONFIG_FILE, system_dir,
                           default_logs_directory, load_runs, time_grid, aligned, run_statistics,
                           cell_distance_to_base)

##### Okabe-Ito, swarm and egocentric keep their colours of comparing_swarm_egocentric.py. #####
##### The lines also differ in dash pattern and the points in marker, so they do not rely #####
##### on colour alone.                                                                    #####
SYSTEM_STYLES = {
    "swarm":                        {"color": "#0072B2", "ls": "-",  "marker": "o"},
    "egocentric":                   {"color": "#D55E00", "ls": "--", "marker": "s"},
    "centralized":                  {"color": "#000000", "ls": (0, (1, 1)), "marker": "X"},
    "naive":                        {"color": "#CC79A7", "ls": (0, (5, 2, 1, 2)), "marker": "h"},
    "expanding_testing/swarm":      {"color": "#009E73", "ls": "-.", "marker": "^"},
    "expanding_testing/egocentric": {"color": "#CC79A7", "ls": ":",  "marker": "D"},
    "MARL/CTDE_MADQN":              {"color": "#E69F00", "ls": (0, (6, 2)),          "marker": "v"},
    "MARL/Independent_MADQN":       {"color": "#56B4E9", "ls": (0, (4, 1, 1, 1, 1, 1)), "marker": "P"},
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

##### Windows the difference of uncertainty to the reference is averaged in, in s. Long #####
##### enough to smooth the recharge waves, short enough to show the first sweep apart.  #####
UNCERTAINTY_WINDOW = 1000.0
REFERENCE_SYSTEM = "naive"

##### Distance bins of the revisit figure, in m: 5 cells #####
DISTANCE_BIN = 100.0
##### A distance is drawn where at least this share of the runs revisited a cell there. The far #####
##### ones only exist in the runs with a base near a corner, a few runs would decide them.       #####
MIN_RUN_SHARE = 0.5


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


def load_systems(systems, logs_directory, transmission_range, number_of_drones, hand_tuned=False):
    loaded = {}
    for system in systems:
        directory = os.path.join(system_dir(system), logs_directory or
                                 default_logs_directory(transmission_range, number_of_drones, hand_tuned))
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


def paired_runs(data, reference_data):
    """
    (row in data, row in reference_data) of every run number both have, when the two are
    seeded alike: run k then has the same base(s) and wind on both. None otherwise.
    """
    if data["config"]["seed"] != reference_data["config"]["seed"]:
        return None
    row_of = {record["run"]: row for row, record in enumerate(reference_data["runs"])}
    pairs = [(row, row_of[record["run"]]) for row, record in enumerate(data["runs"]) if record["run"] in row_of]
    return pairs or None


def bootstrap_relative_difference(values, reference, paired):
    """
    100 * (mean of values / mean of reference - 1), the difference in % of the reference,
    with its percentile bootstrap interval. values and reference have a row per run. Paired,
    row k of both is the same run and the rows are drawn together, so the spread the base
    position adds to both systems cancels out. Unpaired, each system is drawn on its own.
    """
    values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def weights(runs):
        return rng.multinomial(runs, np.full(runs, 1.0 / runs), size=BOOTSTRAP_RESAMPLES) / runs

    values_weights = weights(len(values))
    reference_weights = values_weights if paired else weights(len(reference))
    ratio = ((values_weights @ values.reshape(len(values), -1))
             / (reference_weights @ reference.reshape(len(reference), -1)))
    tail = (100 - CONFIDENCE) / 2
    low, high = np.percentile(100.0 * (ratio - 1.0), [tail, 100 - tail], axis=0)
    mean = 100.0 * (values.mean(axis=0) / reference.mean(axis=0) - 1.0)
    return mean, low.reshape(values.shape[1:]), high.reshape(values.shape[1:])


def plot_uncertainty_difference(loaded, reference, window, steady_state):
    """
    The uncertainty of every system relative to the reference, in %, so differences of a
    few % that the absolute figure hides become readable:
      (a) in time, averaged in windows of window seconds, mean and bootstrap interval;
      (b) the mean over each run from steady_state on, a dot per run, box plot and mean.
    Seeded alike, run k of a system is compared with run k of the reference (same base(s)
    and wind), which removes the spread the base position adds to both. None with a
    single system.
    """
    others = [system for system in loaded if system != reference]
    if not others:
        return None
    reference_data = loaded[reference]
    dt = reference_data["config"]["sample_interval"]

    fig = plt.figure(figsize=(12, 4.6), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=(2.2, 1))
    ax_time, ax_runs = fig.add_subplot(layout[0]), fig.add_subplot(layout[1])
    jitter = np.random.default_rng(JITTER_SEED)
    tested = []
    for position, system in enumerate(others):
        data = loaded[system]
        style = SYSTEM_STYLES[system]
        if data["config"]["sample_interval"] != dt:
            print(f"{system}: sampled every {data['config']['sample_interval']:g} s, {reference} every {dt:g} s, "
                  f"left out of the difference figure")
            continue
        pairs = paired_runs(data, reference_data)
        paired = pairs is not None
        rows, reference_rows = zip(*pairs) if paired else (range(len(data["runs"])),
                                                           range(len(reference_data["runs"])))
        ##### Both on the shorter run, when the durations differ #####
        samples = min(len(data["grid"]), len(reference_data["grid"]))
        grid = reference_data["grid"][:samples]
        uncertainty = np.array([data["series"][row]["global_uncertainty"][:samples] for row in rows])
        reference_uncertainty = np.array([reference_data["series"][row]["global_uncertainty"][:samples]
                                          for row in reference_rows])

        ##### (a) In time #####
        windows, edges = windowed(uncertainty, grid, dt, window)
        reference_windows, _ = windowed(reference_uncertainty, grid, dt, window)
        mean, low, high = bootstrap_relative_difference(windows, reference_windows, paired)
        ax_time.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                             color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax_time.step(edges, np.r_[mean, mean[-1]], where="post", color=style["color"], ls=style["ls"], lw=1.6,
                     label=f"{system}, {len(rows)} runs" + (" paired" if paired else ""))

        ##### (b) Each run, from steady_state on #####
        steady = grid >= steady_state
        run_mean = uncertainty[:, steady].mean(axis=1)
        reference_mean = reference_uncertainty[:, steady].mean(axis=1)
        per_run = 100.0 * (run_mean / (reference_mean if paired else reference_mean.mean()) - 1.0)
        line = {"color": style["color"], "lw": 1.0}
        ax_runs.boxplot(per_run, positions=[position], widths=0.45, showfliers=False, boxprops=line,
                        whiskerprops=line, capprops=line, medianprops={"color": style["color"], "lw": 2.0})
        ax_runs.scatter(position + jitter.uniform(-0.13, 0.13, len(per_run)), per_run, marker=style["marker"],
                        s=18, color=style["color"], alpha=0.6, lw=0, zorder=3)
        total, total_low, total_high = bootstrap_relative_difference(run_mean, reference_mean, paired)
        ax_runs.errorbar(position + 0.36, total, yerr=[[total - total_low], [total_high - total]], fmt="o",
                         color="black", ms=4, capsize=3, lw=1.0, zorder=4)
        summary = (f"{system} vs {reference}: mean uncertainty from {steady_state:g} s {total:+.1f}%, "
                   f"{CONFIDENCE}% interval [{total_low:+.1f}, {total_high:+.1f}] over {len(rows)} runs")
        if paired and len(rows) > 1 and np.any(run_mean != reference_mean):
            p_value = float(wilcoxon(run_mean, reference_mean).pvalue)
            summary += ", paired Wilcoxon " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
            tested.append(f"{system}: " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"))
        print(summary)

    for ax in (ax_time, ax_runs):
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
        ax.grid(True, axis="y", **GRID)
        ax.set_axisbelow(True)
    ax_time.text(0.995, 0.0, f"{reference} (reference) ", transform=ax_time.get_yaxis_transform(),
                 ha="right", va="top", fontsize=7, color="0.35", zorder=5,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    if steady_state > 0:
        ax_time.axvspan(0, steady_state, color=OVER_CAPACITY_COLOR, lw=0, zorder=0)
        ax_time.text(steady_state / 2, 0.02, "left out\nof (b)", transform=ax_time.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7, color="0.35")
    ax_time.grid(True, axis="x", **GRID)
    ax_time.set_xlim(0, reference_data["grid"][-1] + dt)
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel(f"global uncertainty relative\nto {reference} (%)")
    ax_time.set_title(f"(a) In time, {window:g} s windows: mean over the runs, {CONFIDENCE}% bootstrap interval. "
                      f"Below 0: less uncertainty than {reference}", loc="left", fontsize=9)
    ax_time.legend(loc="best", fontsize=8, frameon=False, handlelength=3.5)

    ax_runs.set_xlim(-0.6, len(others) - 0.4)
    ax_runs.set_xticks(range(len(others)))
    ax_runs.set_xticklabels([system.replace("/", "/\n") for system in others], fontsize=8)
    ax_runs.set_ylabel(f"mean uncertainty of the run\nrelative to {reference} (%)")
    ax_runs.set_title("(b) Each run" + (f", from {steady_state:g} s" if steady_state > 0 else "")
                      + ("\npaired Wilcoxon " + ", ".join(tested) if tested else ""), loc="left", fontsize=9)
    ax_runs.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.45", alpha=0.6, ms=4, label="one run"),
                            Line2D([], [], marker="o", ls="none", color="black", ms=4,
                                   label=f"mean, {CONFIDENCE}% interval")],
                   loc="best", fontsize=7, frameon=False, handletextpad=0.3)
    fig.suptitle(f"Global uncertainty relative to {reference}, {scenario(loaded)}")
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


def revisit_by_distance(runs, distances, distance_bin, bins):
    """
    Per distance bin, over every revisit of every run: the mean and the std deviation of
    the time between two visits of the cells at that distance from their nearest base, and
    how many runs revisited a cell there. distances holds the distance of every cell of each
    run, cell_distance_to_base. NaN where there are too few revisits: one for the mean, two
    for the std deviation.
    """
    count, total, squares = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    runs_there = np.zeros(bins, dtype=int)
    for record, distance in zip(runs, distances):
        index = (np.ravel(distance) // distance_bin).astype(int)

        def per_bin(values):
            return np.bincount(index, weights=np.ravel(values).astype(float), minlength=bins)

        run_count = per_bin(record["revisit"]["count"])
        count += run_count
        total += per_bin(record["revisit"]["total"])
        squares += per_bin(record["revisit"]["total_squares"])
        runs_there += run_count > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        variance = np.where(count > 1, (squares - count * mean ** 2) / (count - 1), np.nan)
    ##### Rounding can leave a variance a hair below 0 when every revisit took the same time #####
    return mean, np.sqrt(np.clip(variance, 0.0, None)), runs_there


def plot_revisit(loaded, distance_bin):
    """
    Time between two visits of a cell by the distance from the cell to its nearest base:
    the mean (line) and one std deviation around it (band), over every revisit of every
    run. None when no system has the revisits.
    """
    with_revisit = {}
    for system, data in loaded.items():
        runs = [record for record in data["runs"] if "revisit" in record]
        if len(runs) < len(data["runs"]):
            print(f"{system}: {len(data['runs']) - len(runs)} of {len(data['runs'])} runs have no time between "
                  f"visits (written before it was recorded, or MARL), left out of the revisit figure"
                  + ("" if runs else ", and so is the system"))
        if runs:
            with_revisit[system] = runs
    if not with_revisit:
        return None

    distances = {system: [cell_distance_to_base(record["revisit"]["count"].shape, record["base_positions"])
                          for record in runs] for system, runs in with_revisit.items()}
    bins = int(max(d.max() for ds in distances.values() for d in ds) // distance_bin) + 1
    centres = (np.arange(bins) + 0.5) * distance_bin

    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    drawn = 0.0
    for system, runs in with_revisit.items():
        style = SYSTEM_STYLES[system]
        mean, std, runs_there = revisit_by_distance(runs, distances[system], distance_bin, bins)
        ##### The far distances that only a few runs reach are not drawn, see MIN_RUN_SHARE #####
        mean[runs_there < MIN_RUN_SHARE * len(runs)] = np.nan
        shown = np.isfinite(mean)
        if shown.any():
            drawn = max(drawn, centres[shown].max() + distance_bin / 2)
        ##### The whole map: every bin together #####
        whole_mean, whole_std, _ = revisit_by_distance(runs, [np.zeros_like(d) for d in distances[system]],
                                                       distance_bin, 1)
        ax.fill_between(centres, mean - std, mean + std, where=shown, color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax.plot(centres, mean, color=style["color"], ls=style["ls"], marker=style["marker"], ms=3.5, lw=1.8,
                label=f"{system}, {len(runs)} runs (whole map {whole_mean[0]:.0f} ± {whole_std[0]:.0f} s)")
        ##### Cells seen at most once have no time between two visits, the mean leaves them out #####
        at_most_once = 100.0 * np.mean([np.mean(record["revisit"]["count"] == 0) for record in runs])
        print(f"{system}: time between two visits of a cell {whole_mean[0]:.1f} +- {whole_std[0]:.1f} s over "
              f"every revisit of {len(runs)} runs, {at_most_once:.1f}% of the cells seen at most once (left out)")

    ax.set_xlabel(f"distance from the cell to the nearest base (m), in {distance_bin:g} m bins, "
                  f"where at least {MIN_RUN_SHARE:.0%} of the runs revisited a cell")
    ax.set_ylabel("time between two\nvisits of a cell (s)")
    ax.set_title("Time between two visits: mean over every revisit of the runs, band of ±1 standard deviation",
                 loc="left", fontsize=9)
    ax.set_xlim(0, drawn or bins * distance_bin)
    ax.set_ylim(bottom=0)
    ax.grid(True, **GRID)
    ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=3.5)
    fig.suptitle(f"Time between two visits of a cell by distance to the base, {scenario(loaded)}")
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global uncertainty (mean and std deviation), drones charging above Nc and time between "
                    "two visits of a cell by distance to the base, over many runs")
    parser.add_argument("--system", choices=SYSTEMS + MARL_SYSTEMS, nargs="+", required=True,
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
    parser.add_argument("--distance-bin", type=float, default=DISTANCE_BIN,
                        help="width of the bins of distance to the base the time between two visits "
                             "is grouped in, in m (default: %(default)s)")
    parser.add_argument("--hand-tuned", "--hand_tuned", action="store_true",
                        help="read the runs of multiple_runs.py --hand-tuned, in multiple_runs_hand_tuned/ "
                             "(ignored with --logs-directory)")
    parser.add_argument("--reference",
                        help="system the others are compared to in the difference of uncertainty figure "
                             f"(default: {REFERENCE_SYSTEM} when plotted, otherwise the first --system)")
    parser.add_argument("--uncertainty-window", type=float, default=UNCERTAINTY_WINDOW,
                        help="width of the windows the difference of uncertainty is averaged in, in s "
                             "(default: %(default)s)")
    parser.add_argument("--steady-state", type=float, default=0.0,
                        help="the per-run panel of the difference figure only averages from this time on, "
                             "in s, to leave the first sweep of the map out (default: %(default)s)")
    parser.add_argument("--save",
                        help="also save the figures, as <name>_uncertainty<ext>, <name>_uncertainty_difference<ext>, "
                             "<name>_charging<ext> and <name>_revisit<ext>, the extension sets the format (pdf, png, ...)")
    args = parser.parse_args()
    if args.distance_bin <= 0:
        parser.error("--distance-bin has to be positive")
    if args.uncertainty_window <= 0:
        parser.error("--uncertainty-window has to be positive")

    loaded = load_systems(args.system, args.logs_directory, args.transmission_range, args.drones,
                          args.hand_tuned)
    reference = args.reference or (REFERENCE_SYSTEM if REFERENCE_SYSTEM in loaded else next(iter(loaded)))
    if reference not in loaded:
        parser.error(f"--reference {reference} has no runs here, it has to be one of {', '.join(loaded)}")
    if args.steady_state >= min(data["config"]["duration"] for data in loaded.values()):
        parser.error("--steady-state has to be shorter than the runs")
    figures = {"uncertainty": plot_uncertainty(loaded),
               "uncertainty_difference": plot_uncertainty_difference(loaded, reference, args.uncertainty_window,
                                                                     args.steady_state),
               "charging": plot_charging(loaded, args.window),
               "revisit": plot_revisit(loaded, args.distance_bin)}
    figures = {name: fig for name, fig in figures.items() if fig is not None}
    if args.save:
        stem, extension = os.path.splitext(args.save)
        for name, fig in figures.items():
            path = f"{stem}_{name}{extension or '.pdf'}"
            fig.savefig(path, bbox_inches="tight")
            print(f"{path}: saved")
    plt.show()
