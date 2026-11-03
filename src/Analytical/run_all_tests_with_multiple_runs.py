"""
Runs multiple_runs.py for every TRANSMISSION_RANGE the GA tuned, one range after the
other, so the tuned systems are compared over many runs, each with its recharge
base(s) somewhere else, and not over the single run of all_tests_distance.py.

    python3 Analytical/run_all_tests_with_multiple_runs.py --system swarm                # every range, 10 runs
    python3 Analytical/run_all_tests_with_multiple_runs.py --system egocentric --runs 20 --drones 5 10
    python3 Analytical/run_all_tests_with_multiple_runs.py --system swarm --ranges 100 200
    python3 Analytical/run_all_tests_with_multiple_runs.py --system expanding_testing/swarm --bases 3 \
        --map-size 150 --logs-root multiple_runs_3_bases_150

It can be started from any directory. The ranges are the ones tuned by
all_optimizations_distance.py, found in <system>/ga_logs/ga_logbook_<range>.txt as in
all_tests_distance.py (expanding_testing uses the logbooks of the system it extends),
and multiple_runs.py picks the individual tuned for each one. A range without a
logbook, or without the Parameters line in it, is an error: nothing is run.

Each range and swarm size is written by multiple_runs.py to
<system>/<logs-root>/<range>/<drones>. The default <logs-root> is the directory
multiple_runs.py and plot_multiple_runs.py use with --range and --drones. Runs with
other bases, map size or capacity should go to their own --logs-root, multiple_runs.py
refuses to mix two scenarios in one directory.

Run k of every range is seeded with --seed + k, so every range runs the same bases
and the same wind. The runs already in a directory are skipped by multiple_runs.py,
so a sweep that was interrupted can be started again without repeating them, and
--runs 20 after --runs 10 only runs the new ones. At the end the mean +- std deviation
over the runs of every range is printed and written to
<system>/<logs-root>/ranges_summary.txt.
"""

import os
import sys
import time
import argparse
import subprocess

import numpy as np

from all_tests_distance import tuned_ranges, tuned_individuals, ga_logs_dir
from multiple_runs import (SYSTEMS, NUMBER_OF_RUNS, NUMBER_OF_DRONES, SIMULATION_DURATION,
                           GLOBAL_MAP_SAMPLE_INTERVAL, DEFAULT_LOGS_DIRECTORY, CONFIG_FILE,
                           system_dir, is_expanding, tuned_system, load_runs, time_grid, run_statistics)

ANALYTICAL_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIPLE_RUNS = os.path.join(ANALYTICAL_DIR, "multiple_runs.py")
RANGES_SUMMARY_FILE = "ranges_summary.txt"


def run_range(command: list) -> int:
    """Runs one multiple_runs.py, its output straight to the terminal, and returns its exit status."""
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:
        ##### The Ctrl+C reached multiple_runs.py too. It is left to stop by itself, #####
        ##### killing it could cut short a run file it is writing.                    #####
        process.wait()
        raise


def mean_std(values, decimals: int = 1) -> str:
    """Mean +- std deviation over the runs, a single run has no spread."""
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.{decimals}f} +- {std:.{decimals}f}"


def directory_statistics(directory: str):
    """The number of runs in the directory and their statistics, by name, or None without runs."""
    if not os.path.exists(os.path.join(directory, CONFIG_FILE)):
        return None
    config, runs = load_runs(directory)
    if not runs:
        return None
    grid = time_grid(config)
    statistics = [run_statistics(record, grid) for record in runs]
    return len(runs), {name: [stats[name] for stats in statistics] for name in statistics[0]}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="multiple_runs.py for every tuned transmission range and swarm size, sequentially")
    parser.add_argument("--system", choices=SYSTEMS, required=True,
                        help="which system to run")
    parser.add_argument("--ranges", type=float, nargs="+",
                        help="transmission ranges to run, in m (default: every range in <system>/ga_logs, "
                             "the one of the system it extends for expanding_testing)")
    parser.add_argument("--drones", type=int, nargs="+", default=[NUMBER_OF_DRONES],
                        help="swarm sizes to run, each one on every range (default: %(default)s)")
    parser.add_argument("--logs-root", default=DEFAULT_LOGS_DIRECTORY,
                        help="directory, relative to the system directory, the runs are written to, "
                             "as <range>/<drones> (default: %(default)s)")

    ##### Passed on to every multiple_runs.py, see its help #####
    parser.add_argument("--runs", type=int, default=NUMBER_OF_RUNS,
                        help="runs of every range and swarm size (default: %(default)s)")
    parser.add_argument("--duration", type=int, default=SIMULATION_DURATION,
                        help="simulated seconds of each run (default: %(default)s)")
    parser.add_argument("--sample-interval", type=float, default=GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="run k of every range is seeded with SEED + k (default: %(default)s)")
    parser.add_argument("--bases", type=int, default=1,
                        help="expanding_testing only: recharge bases of each run (default: %(default)s)")
    parser.add_argument("--map-size", type=int, metavar="CELLS",
                        help="expanding_testing only: side of the square map, in cells "
                             "(default: MAP_WIDTH of main.py)")
    parser.add_argument("--charging-capacity", type=int, metavar="NC",
                        help="expanding_testing only: most drones charging at once over all the bases "
                             "(default: int(drones / 3))")
    parser.add_argument("--workers", type=int,
                        help="runs simulated at once (default: one per run, at most one per CPU)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the runs already in every directory and run them all again")
    args = parser.parse_args()

    ##### Checked here, multiple_runs.py would refuse them once per range #####
    if args.runs < 1:
        parser.error("--runs has to be at least 1")
    if not is_expanding(args.system):
        for flag, value in (("--bases", args.bases != 1), ("--map-size", args.map_size is not None),
                            ("--charging-capacity", args.charging_capacity is not None)):
            if value:
                parser.error(f"{flag} is only for the expanding_testing systems, "
                             f"{args.system} has one base on a fixed map")
    if args.bases < 1:
        parser.error("--bases has to be at least 1")

    ranges = args.ranges if args.ranges is not None else list(tuned_ranges(tuned_system(args.system)))
    if not ranges:
        parser.error(f"no ga_logbook_<range>.txt in {ga_logs_dir(tuned_system(args.system))}, nothing to run")
    ##### multiple_runs.py refuses a range without one too, checked here for every range so the #####
    ##### sweep stops before the first one and not after hours of the others.                   #####
    try:
        tuned_individuals(tuned_system(args.system), ranges)
    except ValueError as error:
        parser.error(f"no tuned individual, nothing was run:\n{error}")

    sweep = [(transmission_range, number_of_drones)
             for transmission_range in ranges for number_of_drones in args.drones]
    summary = []
    sweep_start = time.time()
    for done, (transmission_range, number_of_drones) in enumerate(sweep, start=1):
        logs_directory = os.path.join(args.logs_root, f"{transmission_range:g}", str(number_of_drones))
        command = [sys.executable, "-u", MULTIPLE_RUNS,
                   "--system", args.system,
                   "--range", f"{transmission_range:g}",
                   "--drones", str(number_of_drones),
                   "--runs", str(args.runs),
                   "--duration", str(args.duration),
                   "--sample-interval", str(args.sample_interval),
                   "--seed", str(args.seed),
                   "--bases", str(args.bases),
                   "--logs-directory", logs_directory]
        for flag, value in (("--map-size", args.map_size), ("--charging-capacity", args.charging_capacity),
                            ("--workers", args.workers)):
            if value is not None:
                command += [flag, str(value)]
        if args.overwrite:
            command.append("--overwrite")

        print(f"\n##### [{done}/{len(sweep)}] {args.system}: {transmission_range:g} m, "
              f"{number_of_drones} drones, {args.runs} runs #####", flush=True)
        start = time.time()
        try:
            status = run_range(command)
        except KeyboardInterrupt:
            print(f"##### Interrupted during {transmission_range:g} m, {number_of_drones} drones, "
                  f"the remaining ranges were not run #####")
            summary.append((transmission_range, number_of_drones, "interrupted", time.time() - start))
            break
        summary.append((transmission_range, number_of_drones, "ok" if status == 0 else f"failed ({status})",
                        time.time() - start))

    logs_root = os.path.join(system_dir(args.system), args.logs_root)
    lines = [
        f"===== {args.system}: multiple_runs.py on every range, logs in {logs_root}/<range>/<drones> =====",
        f"{args.duration} s, {args.bases} base(s) drawn again on every run, seeds from {args.seed}",
        "Mean +- std deviation over the runs of each range and swarm size",
        "",
        f"{'range':>7} {'drones':>6}  {'result':<12} {'time':>9}  {'runs':>4}  {'mean uncertainty':>18}  "
        f"{'above Nc (%)':>14}  {'dead at the end':>15}  {'cost':>18}",
    ]
    for transmission_range, number_of_drones, result, elapsed in summary:
        row = f"{transmission_range:>5g} m {number_of_drones:>6}  {result:<12} {elapsed / 60:>5.1f} min  "
        statistics = directory_statistics(os.path.join(logs_root, f"{transmission_range:g}", str(number_of_drones)))
        if statistics is None:
            row += f"{'no runs':>4}"
        else:
            runs, values = statistics
            row += (f"{runs:>4}  {mean_std(values['mean_uncertainty']):>18}  {mean_std(values['above_nc']):>14}  "
                    f"{mean_std(values['dead'], 2):>15}  {mean_std(values['cost']):>18}")
        lines.append(row)
    lines.append("")
    lines.append(f"Total: {(time.time() - sweep_start) / 60:.1f} min")
    text = "\n".join(lines)

    print("\n" + text)
    os.makedirs(logs_root, exist_ok=True)
    with open(os.path.join(logs_root, RANGES_SUMMARY_FILE), "w") as f:
        f.write(text + "\n")
    print(f"\nWritten to {os.path.join(logs_root, RANGES_SUMMARY_FILE)}")
    print(f"Plot a range with: python3 Analytical/plot_multiple_runs.py --system {args.system} "
          f"--logs-directory {args.logs_root}/<range>/<drones>")
    return 0 if all(result == "ok" for _, _, result, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
