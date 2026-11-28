"""
Plots the runs multiple_runs.py wrote, for one or more systems, in two figures:

- Global uncertainty: at every time, the mean over the runs (line) and one standard
  deviation around it (band).
- Global uncertainty relative to a reference system (--reference, naive by default), in %,
  when there are several systems: the absolute figure hides differences of a few %. Seeded
  alike, run k of each system is compared with run k of the reference (same base(s) and
  wind), which removes the spread the base position adds to both. (a) in time, a moving
  average over --uncertainty-window seconds; (b) the mean of each run from --steady-state on, a dot
  per run, with a paired Wilcoxon signed-rank test.
- Time on the base, in three panels that aggregate the runs, so they read the same with
  10 runs or 200. Since 2026-11-24 the base has Nc pads and a waiting room
  (recharge_base.py): more than Nc drones never charge at once, the congestion is the
  time the drones wait instead.
  (a) when: the probability of a drone being in the waiting room, the share of the
      drones waiting for a pad, in windows of --window seconds, mean over the runs. Runs
      without a waiting room (before 2026-11-24, MARL) are left out;
  (b) how much a drone waits: the share of its time a drone spends in the waiting room,
      a dot per run over a box plot, and the mean. Two systems seeded alike are compared
      run by run, run k has the same base(s) and wind on both, with a Wilcoxon
      signed-rank test. Runs without a waiting room (before 2026-11-24, MARL) are left out;
  (c) how many at once: the share of the time with k drones on the base, charging and
      waiting, on a log scale. Up to Nc they are all on a pad, right of Nc the other k - Nc
      wait: it shows how often every pad is taken and how long the line gets.
  The intervals are 95% percentile bootstraps of the mean over the runs: they narrow
  as runs are added, the spread between runs (b) does not.
- Drones mapping (since 2026-11-27): the drones neither on the base nor dead, the service the
  swarm delivers. The mean uncertainty follows the mean number in the air, but the same mean
  can be delivered steadily or in waves. (a) in time, the mean in windows of --window seconds;
  (b) the shortfall, the share of the time with at most k drones mapping, for every k, on a log
  scale; (c) how constant it is, the standard deviation in time of the number mapping, a dot
  per run, paired run by run with a Wilcoxon test when seeded alike. (b) and (c) take the time
  from --steady-state on. GOING_TO_BASE is not logged apart: a drone flying to the base counts
  as mapping, its camera still sees the cells on the way.
- Time between two visits of a cell, by the distance from the cell to its nearest base,
  in bins of --distance-bin metres (see multiple_runs.RevisitMonitor for what a visit is):
  the mean (line) and one standard deviation around it (band), over every revisit of the
  cells at that distance in every run. The bases move from run to run, so a distance is
  only drawn where at least half of the runs revisited a cell there. Cells seen at most
  once have no time between two visits and are left out, their share is printed. Runs
  written before the revisits were recorded, and the MARL ones, do not have them.
- Time between two visits relative to the reference (--reference), in %, when there are
  several systems, the counterpart of the uncertainty difference: (a) by distance to the
  base, the mean over every revisit, paired run by run when seeded alike, with a bootstrap
  interval; (b) the whole-map mean of each run, a dot per run, with a paired Wilcoxon
  signed-rank test. Below 0 the cells are seen again sooner than with the reference.

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
import warnings
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
    ##### Only when this script draws expanding_testing/naive itself, e.g. next to naive. #####
    ##### plot_multiple_runs_bases.py draws it with the style of naive.                   #####
    "expanding_testing/naive":      {"color": "#E69F00", "ls": "--", "marker": "h"},
    "MARL/CTDE_MADQN":              {"color": "#E69F00", "ls": (0, (6, 2)),          "marker": "v"},
    "MARL/Independent_MADQN":       {"color": "#56B4E9", "ls": (0, (4, 1, 1, 1, 1, 1)), "marker": "P"},
}
BAND_ALPHA = 0.25
GRID = {"color": "0.92", "lw": 0.5}

##### Windows of (a) of the time on the base, in s. The first recharge wave lasts a few hundred seconds. #####
WINDOW = 50.0
##### Intervals of the means over the runs. Seeded, the figures do not change from call to call. #####
CONFIDENCE = 95
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
JITTER_SEED = 0
OVER_CAPACITY_COLOR = "0.95"

##### Width of the moving average of the difference of uncertainty to the reference, in s. #####
##### Long enough to smooth the recharge waves, short enough to show the first sweep apart. #####
UNCERTAINTY_WINDOW = 1000.0
##### The moving average is read this many times per window, so the line looks continuous #####
MOVING_AVERAGE_POINTS = 20
REFERENCE_SYSTEM = "naive"

##### Distance bins of the revisit figure, in m: 5 cells #####
DISTANCE_BIN = 10.0
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


def moving_average(values, grid, dt, window):
    """
    Mean of every run (values has a row per run) over a window of window seconds centred
    on each instant, read every window / MOVING_AVERAGE_POINTS seconds, and those instants.
    Near the start and the end the window is cut to what the run has.
    """
    samples = values.shape[1]
    half = max(1, int(round(window / dt))) // 2
    centres = np.arange(0, samples, max(1, int(round(window / MOVING_AVERAGE_POINTS / dt))))
    if centres[-1] != samples - 1:
        centres = np.r_[centres, samples - 1]
    low, high = np.maximum(centres - half, 0), np.minimum(centres + half + 1, samples)
    cumulative = np.concatenate([np.zeros((len(values), 1)), np.cumsum(values, axis=1)], axis=1)
    return (cumulative[:, high] - cumulative[:, low]) / (high - low), grid[centres]


def paired_test(values):
    """
    p-value of the Wilcoxon signed-rank test of a value of the runs, run k of one system
    against run k of the other, and how many runs were paired. values is {system: (seed,
    {run: value})}. Seeded alike, run k has the same base(s) and wind on both. None unless
    there are two systems seeded alike, with some run that differs.
    """
    if len(values) != 2:
        return None
    (first_seed, first), (second_seed, second) = values.values()
    if first_seed != second_seed:
        return None
    runs = sorted(set(first) & set(second))
    x, y = (np.array([per_run[run] for run in runs]) for per_run in (first, second))
    if not len(runs) or not np.any(x != y):
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
      (a) in time, a moving average over window seconds, mean and bootstrap interval;
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

        ##### (a) In time, a moving average of each run #####
        smoothed, instants = moving_average(uncertainty, grid, dt, window)
        reference_smoothed, _ = moving_average(reference_uncertainty, grid, dt, window)
        mean, low, high = bootstrap_relative_difference(smoothed, reference_smoothed, paired)
        ax_time.fill_between(instants, low, high, color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax_time.plot(instants, mean, color=style["color"], ls=style["ls"], lw=1.6,
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
    ax_time.set_title(f"(a) In time, {window:g} s moving average: mean over the runs, {CONFIDENCE}% bootstrap interval. "
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


def pads(data):
    """
    Pads of the runs, over every base: Nc per base. expanding_testing gives every base the
    capacity it is run with (PADS_PER_BASE is None), so B bases have B * Nc.
    """
    return sorted({record["nc_charging_base"] * len(record["base_positions"]) for record in data["runs"]})


def has_waiting(data):
    """Runs written before the waiting room (2026-11-24), and the MARL ones, have no waiting series."""
    return all("waiting" in series for series in data["series"])


def plot_charging(loaded, window):
    ##### When, how much and how many at once: every panel aggregates the runs, so it reads #####
    ##### the same with 10 runs or 200, and the intervals narrow as runs are added.         #####
    fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=(1, 1.25), width_ratios=(1, 1.35))
    plot_waiting_in_time(fig.add_subplot(layout[0, :]), loaded, window)
    plot_waiting_per_run(fig.add_subplot(layout[1, 0]), loaded)
    plot_on_base_at_once(fig.add_subplot(layout[1, 1]), loaded)
    fig.suptitle(f"Time on the base over the runs, {scenario(loaded)}")
    return fig


def plot_waiting_in_time(ax, loaded, window):
    """
    (a) The probability of a drone waiting in the waiting room, the share of the drones
    waiting for a pad, in windows of window seconds. Runs without a waiting room are left out.
    """
    title = (f"(a) When: probability of a drone in the waiting room, in {window:g} s windows "
             f"(mean over the runs, {CONFIDENCE}% bootstrap interval)")
    drawn = False
    for system, data in loaded.items():
        drones = data["config"]["drones"]
        ##### Still printed: the panel no longer shows the drones charging #####
        charging = np.array([series["charging"] / drones for series in data["series"]])
        whole, whole_low, whole_high = bootstrap_mean(100.0 * charging.mean(axis=1))
        print(f"{system}: a drone is charging {whole:.2f}% of the time, {CONFIDENCE}% interval "
              f"[{whole_low:.2f}, {whole_high:.2f}] over {len(charging)} runs")
        if not has_waiting(data):
            print(f"{system}: runs without a waiting room (before 2026-11-24, or MARL), "
                  f"left out of the waiting in time panel")
            continue
        style = SYSTEM_STYLES[system]
        waiting = np.array([series["waiting"] / drones for series in data["series"]])
        share, edges = windowed(waiting, data["grid"], data["config"]["sample_interval"], window)
        mean, low, high = bootstrap_mean(100.0 * share)
        ##### The last value is repeated so the step covers the last window too #####
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                        color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax.step(edges, np.r_[mean, mean[-1]], where="post", color=style["color"], ls=style["ls"], lw=1.4,
                label=f"{system}, {len(data['runs'])} runs")
        drawn = True

    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, max(data["config"]["duration"] for data in loaded.values()))
    if not drawn:
        ax.text(0.5, 0.5, "no waiting room in these runs\n(written before 2026-11-24, or MARL)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.35")
        ax.set_yticks([])
        return
    ax.set_ylabel("P(a drone is waiting) (%)")
    ax.set_ylim(0, min(100, ax.get_ylim()[1]))
    ax.grid(True, **GRID)
    ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=3.5)


def plot_waiting_per_run(ax, loaded):
    """(b) The share of its time a drone waits, each run: one dot per run over a box plot, and the mean."""
    waiting = {}
    for system, data in loaded.items():
        if not has_waiting(data):
            print(f"{system}: runs without a waiting room (before 2026-11-24, or MARL), "
                  f"left out of the waiting panel")
            continue
        drones = data["config"]["drones"]
        ##### Share of the drone*time spent waiting, in %: the mean drones waiting over the drones #####
        waiting[system] = {record["run"]: 100.0 * float(series["waiting"].mean()) / drones
                           for record, series in zip(data["runs"], data["series"])}
        ##### The same waits over the time on the ground, charging or waiting: how much of a stop #####
        ##### at the base goes to the line                                                         #####
        ground = sum(float(series["waiting"].sum() + series["charging"].sum()) for series in data["series"])
        on_ground = (100.0 * sum(float(series["waiting"].sum()) for series in data["series"]) / ground
                     if ground else 0.0)
        print(f"{system}: waiting is {on_ground:.1f}% of the time on the base")

    if not waiting:
        ax.text(0.5, 0.5, "no waiting room in these runs\n(written before 2026-11-24, or MARL)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.35")
        ax.set_xticks([])
        ax.set_title("(b) How much a drone waits: each run", loc="left", fontsize=9)
        return

    jitter = np.random.default_rng(JITTER_SEED)
    positions = np.arange(len(waiting))
    for position, (system, per_run) in zip(positions, waiting.items()):
        style = SYSTEM_STYLES[system]
        shares = np.array(list(per_run.values()))
        line = {"color": style["color"], "lw": 1.0}
        ax.boxplot(shares, positions=[position], widths=0.45, showfliers=False, boxprops=line,
                   whiskerprops=line, capprops=line, medianprops={"color": style["color"], "lw": 2.0})
        ax.scatter(position + jitter.uniform(-0.13, 0.13, len(shares)), shares, marker=style["marker"],
                   s=16, color=style["color"], alpha=0.55, lw=0, zorder=3)
        mean, low, high = bootstrap_mean(shares)
        ax.errorbar(position + 0.36, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black",
                    ms=4, capsize=3, lw=1.0, zorder=4)
        print(f"{system}: a drone waits {mean:.3f}% of the time, {CONFIDENCE}% interval "
              f"[{low:.3f}, {high:.3f}] over {len(shares)} runs")

    title = "(b) How much a drone waits: each run"
    test = paired_test({system: (loaded[system]["config"]["seed"], per_run) for system, per_run in waiting.items()})
    if test is not None:
        p_value, pairs = test
        compared = ("paired Wilcoxon signed-rank " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
                    + f", {pairs} runs")
        title += "\n" + compared
        print(f"{' vs '.join(waiting)}, waiting: {compared}")
    top = max(max(per_run.values()) for per_run in waiting.values())
    ax.set_ylim(0, 1.12 * top if top > 0 else 1.0)
    ax.set_xlim(-0.6, len(waiting) - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([system.replace("/", "/\n") for system in waiting], fontsize=8)
    ax.set_ylabel("time a drone waits for a pad\n(% of its time)")
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.45", alpha=0.55, ms=4, label="one run"),
                       Line2D([], [], marker="o", ls="none", color="black", ms=4,
                              label=f"mean, {CONFIDENCE}% interval")],
              loc="upper left", fontsize=7, frameon=False, handletextpad=0.3)


def plot_on_base_at_once(ax, loaded):
    """
    (c) The share of the time with k drones on the base, charging and waiting. Up to Nc
    they all charge, right of Nc the other k - Nc wait. Runs without a waiting room only
    count the drones charging, which then could go above Nc.
    """
    drones = max(data["config"]["drones"] for data in loaded.values())
    counts = np.arange(drones + 1)
    for index, (system, data) in enumerate(loaded.items()):
        style = SYSTEM_STYLES[system]
        on_base = [(series["charging"] + series["waiting"] if "waiting" in series else series["charging"]).astype(int)
                   for series in data["series"]]
        occupancy = 100.0 * np.array([np.bincount(count, minlength=drones + 1) for count in on_base]) / len(data["grid"])
        mean, low, high = bootstrap_mean(occupancy)
        ##### Never seen in any run: nothing to draw on a log scale #####
        seen = mean > 0
        ##### Side by side, so the intervals of the systems do not hide each other #####
        offset = (index - (len(loaded) - 1) / 2) * 0.12
        ax.errorbar(counts[seen] + offset, mean[seen], yerr=[(mean - low)[seen], (high - mean)[seen]],
                    marker=style["marker"], ls=style["ls"], color=style["color"], ms=5, lw=1.2, capsize=2,
                    label=system)
        full = min(pads(data))
        print(f"{system}: every pad taken {mean[full:].sum():.2f}% of the time, "
              f"some drone waiting {mean[full + 1:].sum():.2f}%")

    nc = sorted({count for data in loaded.values() for count in pads(data)})
    ax.axvspan(nc[0] + 0.5, drones + 0.5, color=OVER_CAPACITY_COLOR, lw=0, zorder=0)
    for value in nc:
        ax.axvline(value + 0.5, color="0.35", ls=":", lw=1.0)
        ax.text(value + 0.58, 0.03, f"{value} pads taken, the rest wait", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xticks(counts)
    ax.set_xlim(-0.5, drones + 0.5)
    ax.set_xlabel("drones on the base at once (k), charging and waiting")
    ax.set_ylabel("time with k on the base\n(% of the run, log scale)")
    ax.set_title("(c) How many at once: occupancy of the base and its line", loc="left", fontsize=9)
    ax.grid(True, axis="y", which="major", **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=3.5)


def drones_mapping(data, series):
    """
    Drones mapping at every sample of a run: the drones neither on the base (charging or waiting)
    nor dead. The runs do not log GOING_TO_BASE apart, so a drone flying to the base is counted
    as mapping: its camera still sees the cells under it on the way.
    """
    on_base = series["charging"] + (series["waiting"] if "waiting" in series else 0)
    return data["config"]["drones"] - on_base - series["dead"]


def plot_mapping(loaded, window, steady_state):
    """
    How many drones are mapping, the service the swarm delivers. The mean uncertainty follows
    the mean number of drones in the air, but two systems with the same mean can deliver it
    differently: always N - 2, or N most of the time and N - 4 in the recharge waves.
      (a) in time: the mean number mapping in windows of window seconds, mean over the runs;
      (b) the shortfall: the share of the time with at most k drones mapping, for every k;
      (c) how constant it is: the standard deviation in time of the number mapping, a dot per
          run, paired run by run (Wilcoxon) when seeded alike.
    (b) and (c) only take the time from steady_state on.
    """
    fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=(1, 1.25), width_ratios=(1.35, 1))
    ax_time, ax_short, ax_runs = (fig.add_subplot(layout[0, :]), fig.add_subplot(layout[1, 0]),
                                  fig.add_subplot(layout[1, 1]))
    drones = max(data["config"]["drones"] for data in loaded.values())
    counts = np.arange(drones)
    jitter = np.random.default_rng(JITTER_SEED)
    variability = {}
    for index, (system, data) in enumerate(loaded.items()):
        style = SYSTEM_STYLES[system]
        dt = data["config"]["sample_interval"]
        mapping = np.array([drones_mapping(data, series) for series in data["series"]], dtype=float)
        steady = data["grid"] >= steady_state

        ##### (a) In time #####
        share, edges = windowed(mapping, data["grid"], dt, window)
        mean, low, high = bootstrap_mean(share)
        whole, whole_low, whole_high = bootstrap_mean(mapping[:, steady].mean(axis=1))
        ax_time.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                             color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax_time.step(edges, np.r_[mean, mean[-1]], where="post", color=style["color"], ls=style["ls"], lw=1.4,
                     label=f"{system}, {len(data['runs'])} runs, {whole:.2f} mapping on average")

        ##### (b) The shortfall: share of the time with at most k drones mapping #####
        at_most = 100.0 * np.array([[np.mean(run[steady] <= k) for k in counts] for run in mapping])
        short, short_low, short_high = bootstrap_mean(at_most)
        seen = short > 0
        offset = (index - (len(loaded) - 1) / 2) * 0.12
        ax_short.errorbar(counts[seen] + offset, short[seen],
                          yerr=[(short - short_low)[seen], (short_high - short)[seen]],
                          marker=style["marker"], ls=style["ls"], color=style["color"], ms=5, lw=1.2, capsize=2,
                          label=system)

        ##### (c) How constant: the standard deviation in time of each run #####
        spread_in_time = mapping[:, steady].std(axis=1)
        variability[system] = {record["run"]: float(value) for record, value in zip(data["runs"], spread_in_time)}
        line = {"color": style["color"], "lw": 1.0}
        ax_runs.boxplot(spread_in_time, positions=[index], widths=0.45, showfliers=False, boxprops=line,
                        whiskerprops=line, capprops=line, medianprops={"color": style["color"], "lw": 2.0})
        ax_runs.scatter(index + jitter.uniform(-0.13, 0.13, len(spread_in_time)), spread_in_time,
                        marker=style["marker"], s=16, color=style["color"], alpha=0.55, lw=0, zorder=3)
        spread_mean, spread_low, spread_high = bootstrap_mean(spread_in_time)
        ax_runs.errorbar(index + 0.36, spread_mean, yerr=[[spread_mean - spread_low], [spread_high - spread_mean]],
                         fmt="o", color="black", ms=4, capsize=3, lw=1.0, zorder=4)

        since = f" from {steady_state:g} s" if steady_state > 0 else ""
        nc = min(pads(data))
        print(f"{system}: {whole:.3f} drones mapping on average{since}, {CONFIDENCE}% interval "
              f"[{whole_low:.3f}, {whole_high:.3f}] over {len(mapping)} runs; std in time {spread_mean:.3f} "
              f"[{spread_low:.3f}, {spread_high:.3f}]")
        print(f"{system}: share of the time{since} with at most k drones mapping: "
              + ", ".join(f"k={k} {short[k]:.2f}%" for k in counts if short[k] > 0)
              + f" (at most {drones - nc - 1}: some drone waiting or dead)")

    ##### (a) #####
    for data in loaded.values():
        nc = min(pads(data))
        n = data["config"]["drones"]
        ax_time.axhline(n - nc, color="0.35", ls=":", lw=1.0)
        ax_time.text(0.995, n - nc, f"{n} - {nc}: every pad taken, nobody waiting ", ha="right", va="bottom",
                     fontsize=7, color="0.35", transform=ax_time.get_yaxis_transform(), zorder=5,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        break
    if steady_state > 0:
        ax_time.axvspan(0, steady_state, color=OVER_CAPACITY_COLOR, lw=0, zorder=0)
        ax_time.text(steady_state / 2, 0.02, "left out\nof (b), (c)", transform=ax_time.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7, color="0.35")
    ax_time.set_xlim(0, max(data["config"]["duration"] for data in loaded.values()))
    ax_time.set_ylim(top=drones + 0.2)
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("drones mapping")
    ax_time.set_title(f"(a) In time: mean number of drones mapping in {window:g} s windows (mean over the runs, "
                      f"{CONFIDENCE}% bootstrap interval)", loc="left", fontsize=9)
    ax_time.grid(True, **GRID)
    ax_time.legend(loc="lower right", fontsize=8, frameon=False, handlelength=3.5)

    ##### (b) #####
    nc = sorted({count for data in loaded.values() for count in pads(data)})
    ax_short.axvspan(-0.5, drones - nc[0] - 0.5, color=OVER_CAPACITY_COLOR, lw=0, zorder=0)
    ax_short.text(-0.4, 0.97, "shaded: some drone waits\nfor a pad (or is dead)",
                  transform=ax_short.get_xaxis_transform(), ha="left", va="top", fontsize=7)
    ax_short.set_yscale("log")
    ax_short.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_short.set_xticks(counts)
    ax_short.set_xlim(-0.5, drones - 0.5)
    ax_short.set_xlabel("at most k drones mapping")
    ax_short.set_ylabel("time with at most k mapping\n(% of the run, log scale)")
    ax_short.set_title("(b) Shortfall: how often the service drops to k or fewer"
                       + (f", from {steady_state:g} s" if steady_state > 0 else ""), loc="left", fontsize=9)
    ax_short.grid(True, axis="y", which="major", **GRID)
    ax_short.set_axisbelow(True)
    ax_short.legend(loc="lower right", fontsize=8, frameon=False, handlelength=3.5)

    ##### (c) #####
    title = "(c) How constant: std in time of\nthe drones mapping, each run" + (
        f", from {steady_state:g} s" if steady_state > 0 else "")
    test = paired_test({system: (loaded[system]["config"]["seed"], per_run) for system, per_run in variability.items()})
    if test is not None:
        p_value, pairs = test
        compared = ("paired Wilcoxon " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
                    + f", {pairs} runs")
        title += "\n" + compared
        print(f"{' vs '.join(variability)}, std in time of the drones mapping: {compared}")
    ax_runs.set_xlim(-0.6, len(variability) - 0.4)
    ax_runs.set_xticks(range(len(variability)))
    ax_runs.set_xticklabels([system.replace("/", "/\n") for system in variability], fontsize=8)
    ax_runs.set_ylim(bottom=0)
    ax_runs.set_ylabel("std in time of the drones mapping")
    ax_runs.set_title(title, loc="left", fontsize=9)
    ax_runs.grid(True, axis="y", **GRID)
    ax_runs.set_axisbelow(True)
    ax_runs.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.45", alpha=0.55, ms=4, label="one run"),
                            Line2D([], [], marker="o", ls="none", color="black", ms=4,
                                   label=f"mean, {CONFIDENCE}% interval")],
                   loc="lower left", fontsize=7, frameon=False, handletextpad=0.3)
    fig.suptitle(f"Drones mapping over the runs, {scenario(loaded)}")
    return fig


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


def revisit_sums(record, distance_bin, bins):
    """
    Number of revisits and their summed time per distance bin (distance from the cell to its
    nearest base), for one run: the pooled mean of a bin over some runs is the sum of their
    times over the sum of their counts, the mean over every revisit, as in plot_revisit.
    """
    distance = cell_distance_to_base(record["revisit"]["count"].shape, record["base_positions"])
    index = (np.ravel(distance) // distance_bin).astype(int)
    count = np.bincount(index, weights=np.ravel(record["revisit"]["count"]).astype(float), minlength=bins)
    total = np.bincount(index, weights=np.ravel(record["revisit"]["total"]).astype(float), minlength=bins)
    return count[:bins], total[:bins]


def bootstrap_relative_pooled(counts, totals, reference_counts, reference_totals, paired):
    """
    100 * (pooled mean time between two visits / pooled mean of the reference - 1), per
    column, with its percentile bootstrap interval. counts and totals have a row per run.
    Paired, row k of both is the same run (same base(s) and wind) and the rows are drawn
    together, as in bootstrap_relative_difference. NaN where either side has no revisit.
    """
    counts, totals = np.asarray(counts, dtype=float), np.asarray(totals, dtype=float)
    reference_counts = np.asarray(reference_counts, dtype=float)
    reference_totals = np.asarray(reference_totals, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def weights(runs):
        return rng.multinomial(runs, np.full(runs, 1.0 / runs), size=BOOTSTRAP_RESAMPLES) / runs

    values_weights = weights(len(counts))
    reference_weights = values_weights if paired else weights(len(reference_counts))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = ((values_weights @ totals) / (values_weights @ counts)
                 / ((reference_weights @ reference_totals) / (reference_weights @ reference_counts)))
        mean = 100.0 * ((totals.sum(axis=0) / counts.sum(axis=0))
                        / (reference_totals.sum(axis=0) / reference_counts.sum(axis=0)) - 1.0)
    tail = (100 - CONFIDENCE) / 2
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    ##### A bin no resample reaches is all NaN: nanpercentile warns and gives NaN, which is what we want #####
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(100.0 * (ratio - 1.0), [tail, 100 - tail], axis=0)
    mean = np.where(np.isfinite(mean), mean, np.nan)
    return mean, low, high


def plot_revisit_difference(loaded, reference, distance_bin):
    """
    The time between two visits of a cell of every system relative to the reference, in %,
    the counterpart of plot_uncertainty_difference for the revisits. Below 0 the cells are
    seen again sooner than with the reference.
      (a) by distance from the cell to its nearest base, in bins of distance_bin metres: the
          pooled mean over every revisit of the runs, and its bootstrap interval. A bin is
          drawn where at least MIN_RUN_SHARE of the runs of both systems revisited a cell;
      (b) the whole map, a dot per run (the mean over every revisit of the run, relative to
          run k of the reference when seeded alike), box plot, pooled mean and interval,
          paired Wilcoxon signed-rank test.
    The revisits are summed over the whole run, there is no steady-state cut. Cells seen at
    most once have no time between two visits and are left out of both systems. None with a
    single system, or when the reference or every other system has no revisits.
    """
    def with_revisit(system):
        return [row for row, record in enumerate(loaded[system]["runs"]) if "revisit" in record]

    reference_data = loaded[reference]
    reference_rows_all = with_revisit(reference)
    others = [system for system in loaded if system != reference and with_revisit(system)]
    if not others or not reference_rows_all:
        return None

    ##### Distance bins shared by every system #####
    largest = max(cell_distance_to_base(record["revisit"]["count"].shape, record["base_positions"]).max()
                  for data in loaded.values() for record in data["runs"] if "revisit" in record)
    bins = int(largest // distance_bin) + 1
    centres = (np.arange(bins) + 0.5) * distance_bin

    fig = plt.figure(figsize=(12, 4.6), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=(2.2, 1))
    ax_distance, ax_runs = fig.add_subplot(layout[0]), fig.add_subplot(layout[1])
    jitter = np.random.default_rng(JITTER_SEED)
    tested = []
    drawn = 0.0
    for position, system in enumerate(others):
        data = loaded[system]
        style = SYSTEM_STYLES[system]
        rows_all = with_revisit(system)
        pairs = paired_runs(data, reference_data)
        if pairs is not None:
            pairs = [(row, reference_row) for row, reference_row in pairs
                     if row in rows_all and reference_row in reference_rows_all]
        paired = bool(pairs)
        rows, reference_rows = zip(*pairs) if paired else (rows_all, reference_rows_all)

        sums = [revisit_sums(data["runs"][row], distance_bin, bins) for row in rows]
        reference_sums = [revisit_sums(reference_data["runs"][row], distance_bin, bins) for row in reference_rows]
        counts, totals = np.array([s[0] for s in sums]), np.array([s[1] for s in sums])
        reference_counts = np.array([s[0] for s in reference_sums])
        reference_totals = np.array([s[1] for s in reference_sums])

        ##### (a) By distance. The far bins that only a few runs reach are not drawn, see MIN_RUN_SHARE #####
        mean, low, high = bootstrap_relative_pooled(counts, totals, reference_counts, reference_totals, paired)
        enough = (((counts > 0).sum(axis=0) >= MIN_RUN_SHARE * len(rows))
                  & ((reference_counts > 0).sum(axis=0) >= MIN_RUN_SHARE * len(reference_rows)))
        mean, low, high = (np.where(enough, value, np.nan) for value in (mean, low, high))
        shown = np.isfinite(mean)
        if shown.any():
            drawn = max(drawn, centres[shown].max() + distance_bin / 2)
        ax_distance.fill_between(centres, low, high, where=shown & np.isfinite(low) & np.isfinite(high),
                                 color=style["color"], alpha=BAND_ALPHA, lw=0)
        ax_distance.plot(centres, mean, color=style["color"], ls=style["ls"], marker=style["marker"], ms=3.5,
                         lw=1.6, label=f"{system}, {len(rows)} runs" + (" paired" if paired else ""))

        ##### (b) The whole map, each run #####
        run_mean = totals.sum(axis=1) / np.maximum(counts.sum(axis=1), 1)
        reference_mean = reference_totals.sum(axis=1) / np.maximum(reference_counts.sum(axis=1), 1)
        per_run = 100.0 * (run_mean / (reference_mean if paired else reference_mean.mean()) - 1.0)
        line = {"color": style["color"], "lw": 1.0}
        ax_runs.boxplot(per_run, positions=[position], widths=0.45, showfliers=False, boxprops=line,
                        whiskerprops=line, capprops=line, medianprops={"color": style["color"], "lw": 2.0})
        ax_runs.scatter(position + jitter.uniform(-0.13, 0.13, len(per_run)), per_run, marker=style["marker"],
                        s=18, color=style["color"], alpha=0.6, lw=0, zorder=3)
        total, total_low, total_high = bootstrap_relative_pooled(counts.sum(axis=1, keepdims=True),
                                                                 totals.sum(axis=1, keepdims=True),
                                                                 reference_counts.sum(axis=1, keepdims=True),
                                                                 reference_totals.sum(axis=1, keepdims=True), paired)
        total, total_low, total_high = float(total[0]), float(total_low[0]), float(total_high[0])
        ax_runs.errorbar(position + 0.36, total, yerr=[[total - total_low], [total_high - total]], fmt="o",
                         color="black", ms=4, capsize=3, lw=1.0, zorder=4)
        at_most_once = 100.0 * np.mean([np.mean(data["runs"][row]["revisit"]["count"] == 0) for row in rows])
        reference_once = 100.0 * np.mean([np.mean(reference_data["runs"][row]["revisit"]["count"] == 0)
                                          for row in reference_rows])
        summary = (f"{system} vs {reference}: time between two visits over the whole map {total:+.1f}%, "
                   f"{CONFIDENCE}% interval [{total_low:+.1f}, {total_high:+.1f}] over {len(rows)} runs "
                   f"(cells seen at most once, left out: {at_most_once:.1f}% vs {reference_once:.1f}%)")
        if paired and len(rows) > 1 and np.any(run_mean != reference_mean):
            p_value = float(wilcoxon(run_mean, reference_mean).pvalue)
            summary += ", paired Wilcoxon " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
            tested.append(f"{system}: " + ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"))
        print(summary)

    for ax in (ax_distance, ax_runs):
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
        ax.grid(True, axis="y", **GRID)
        ax.set_axisbelow(True)
    ax_distance.text(0.995, 0.0, f"{reference} (reference) ", transform=ax_distance.get_yaxis_transform(),
                     ha="right", va="top", fontsize=7, color="0.35", zorder=5,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax_distance.grid(True, axis="x", **GRID)
    ax_distance.set_xlim(0, drawn or bins * distance_bin)
    ax_distance.set_xlabel(f"distance from the cell to the nearest base (m), in {distance_bin:g} m bins, "
                           f"where at least {MIN_RUN_SHARE:.0%} of the runs revisited a cell")
    ax_distance.set_ylabel(f"time between two visits\nrelative to {reference} (%)")
    ax_distance.set_title(f"(a) By distance to the base: mean over every revisit of the runs, {CONFIDENCE}% "
                          f"bootstrap interval\nBelow 0: cells seen again sooner than with {reference}",
                          loc="left", fontsize=9)
    ax_distance.legend(loc="best", fontsize=8, frameon=False, handlelength=3.5)

    ax_runs.set_xlim(-0.6, len(others) - 0.4)
    ax_runs.set_xticks(range(len(others)))
    ax_runs.set_xticklabels([system.replace("/", "/\n") for system in others], fontsize=8)
    ax_runs.set_ylabel(f"mean time between two visits\nof the run relative to {reference} (%)")
    ax_runs.set_title("(b) Each run, whole map" + ("\npaired Wilcoxon\n" + "\n".join(tested) if tested else ""),
                      loc="left", fontsize=9)
    ax_runs.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.45", alpha=0.6, ms=4, label="one run"),
                            Line2D([], [], marker="o", ls="none", color="black", ms=4,
                                   label=f"mean, {CONFIDENCE}% interval")],
                   loc="best", fontsize=7, frameon=False, handletextpad=0.3)
    fig.suptitle(f"Time between two visits of a cell relative to {reference}, {scenario(loaded)}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global uncertainty (mean and std deviation), time charging and waiting on the base and time between "
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
                        help="width of the windows the probability of a drone waiting for a pad, and the "
                             "mean number of drones mapping, are estimated in, in s (default: %(default)s)")
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
                        help="width of the moving average of the difference of uncertainty, in s "
                             "(default: %(default)s)")
    parser.add_argument("--steady-state", type=float, default=0.0,
                        help="the per-run panel of the difference figure, and panels (b) and (c) of the drones "
                             "mapping, only take the time from this time on, in s, to leave the first sweep of "
                             "the map out (default: %(default)s)")
    parser.add_argument("--save",
                        help="also save the figures, as <name>_uncertainty<ext>, <name>_uncertainty_difference<ext>, "
                             "<name>_charging<ext>, <name>_mapping<ext>, <name>_revisit<ext> and "
                             "<name>_revisit_difference<ext>, the extension sets the format (pdf, png, ...)")
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
               "mapping": plot_mapping(loaded, args.window, args.steady_state),
               "revisit": plot_revisit(loaded, args.distance_bin),
               "revisit_difference": plot_revisit_difference(loaded, reference, args.distance_bin)}
    figures = {name: fig for name, fig in figures.items() if fig is not None}
    if args.save:
        stem, extension = os.path.splitext(args.save)
        for name, fig in figures.items():
            path = f"{stem}_{name}{extension or '.pdf'}"
            fig.savefig(path, bbox_inches="tight")
            print(f"{path}: saved")
    plt.show()
