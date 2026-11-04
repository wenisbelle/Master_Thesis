"""
Runs multiple_runs_bases.py for every TRANSMISSION_RANGE the GA tuned, one range after
the other, so the tuned systems are compared on the multi-base scenarios over many
runs, each with its bases in random quadrants. It is
Analytical/run_all_tests_with_multiple_runs.py for expanding_testing.

    cd src
    export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm egocentric
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm --ranges 100 200
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system egocentric \
        --scenario 100x100_10drones_2bases 100x100_10drones_3bases --runs 40

It can be started from any directory. The ranges are the ones tuned by
all_optimizations_distance.py, found in Analytical/<system>/ga_logs/ga_logbook_<range>.txt,
for every system given, and each system runs with the best individual of its own
logbook, the "Parameters:" line. A range without a logbook for one of the systems, or
without the Parameters line in it, is an error: nothing is run.

Every range runs every scenario of multiple_runs_bases.py (all four by default) on
every system, in one call to it, so all of its runs share one pool of processes. They
are written to <system>/<logs-root>/<range>/<scenario>, here in expanding_testing.

Run k of every range and scenario is seeded with --seed + k, so every range runs the
same bases and the same wind. The runs already in a directory are skipped by
multiple_runs_bases.py, so a sweep that was interrupted can be started again without
repeating them, and --runs 40 after --runs 20 only runs the new ones. At the end the
mean +- std deviation over the runs of every range and scenario is printed and written
to <system>/<logs-root>/ranges_summary.txt.
"""

import os
import sys
import time
import argparse

EXPANDING_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICAL_DIR = os.path.dirname(EXPANDING_DIR)
if ANALYTICAL_DIR not in sys.path:
    sys.path.insert(0, ANALYTICAL_DIR)

from all_tests_distance import tuned_ranges, tuned_individuals, ga_logs_dir
from multiple_runs import SIMULATION_DURATION, GLOBAL_MAP_SAMPLE_INTERVAL, system_dir, tuned_system
from run_all_tests_with_multiple_runs import RANGES_SUMMARY_FILE, run_range, mean_std, directory_statistics
from multiple_runs_bases import SYSTEMS, SCENARIOS, NUMBER_OF_RUNS, DEFAULT_LOGS_ROOT, scenario_directory

MULTIPLE_RUNS_BASES = os.path.join(EXPANDING_DIR, "multiple_runs_bases.py")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="multiple_runs_bases.py for every tuned transmission range, sequentially")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", required=True,
                        help="system(s) to run, from expanding_testing")
    parser.add_argument("--ranges", type=float, nargs="+",
                        help="transmission ranges to run, in m (default: every range in "
                             "Analytical/<system>/ga_logs, of every system given)")
    parser.add_argument("--scenario", choices=SCENARIOS, nargs="+", default=list(SCENARIOS),
                        help="scenario(s) to run on every range (default: all of them)")
    parser.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT,
                        help="directory, relative to each system directory, the runs are written to, "
                             "as <range>/<scenario> (default: %(default)s)")

    ##### Passed on to every multiple_runs_bases.py, see its help #####
    parser.add_argument("--runs", type=int, default=NUMBER_OF_RUNS,
                        help="runs of every system, range and scenario (default: %(default)s)")
    parser.add_argument("--duration", type=int, default=SIMULATION_DURATION,
                        help="simulated seconds of each run (default: %(default)s)")
    parser.add_argument("--sample-interval", type=float, default=GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="run k of every range is seeded with SEED + k (default: %(default)s)")
    parser.add_argument("--workers", type=int,
                        help="runs simulated at once (default: one per run, at most one per CPU)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the runs already in every directory and run them all again")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs has to be at least 1")
    systems = [SYSTEMS[name] for name in dict.fromkeys(args.system)]
    scenarios = list(dict.fromkeys(args.scenario))

    if args.ranges is not None:
        ranges = sorted(set(args.ranges))
    else:
        ##### Every range tuned for any of the systems: one the other system lacks is an error below #####
        ranges = sorted({r for system in systems for r in tuned_ranges(tuned_system(system))})
    if not ranges:
        parser.error("no ga_logbook_<range>.txt in " +
                     ", ".join(ga_logs_dir(tuned_system(system)) for system in systems) + ", nothing to run")
    ##### multiple_runs_bases.py refuses a range without one too, checked here for every range #####
    ##### and system, so the sweep stops before the first one and not after hours of the others. #####
    errors = []
    for system in systems:
        try:
            tuned_individuals(tuned_system(system), ranges)
        except ValueError as error:
            errors.append(f"{tuned_system(system)}:\n{error}")
    if errors:
        parser.error("no tuned individual, nothing was run:\n" + "\n".join(errors))

    summary = []
    sweep_start = time.time()
    for done, transmission_range in enumerate(ranges, start=1):
        command = [sys.executable, "-u", MULTIPLE_RUNS_BASES,
                   "--system", *args.system,
                   "--scenario", *scenarios,
                   "--range", f"{transmission_range:g}",
                   "--runs", str(args.runs),
                   "--duration", str(args.duration),
                   "--sample-interval", str(args.sample_interval),
                   "--seed", str(args.seed),
                   "--logs-root", args.logs_root]
        if args.workers is not None:
            command += ["--workers", str(args.workers)]
        if args.overwrite:
            command.append("--overwrite")

        print(f"\n##### [{done}/{len(ranges)}] {transmission_range:g} m: {', '.join(args.system)}, "
              f"{len(scenarios)} scenario(s), {args.runs} runs each #####", flush=True)
        start = time.time()
        try:
            status = run_range(command)
        except KeyboardInterrupt:
            print(f"##### Interrupted during {transmission_range:g} m, the remaining ranges were not run #####")
            summary.append((transmission_range, "interrupted", time.time() - start))
            break
        summary.append((transmission_range, "ok" if status == 0 else f"failed ({status})", time.time() - start))

    total = (time.time() - sweep_start) / 60
    texts = []
    for system in systems:
        logs_root = os.path.join(system_dir(system), args.logs_root)
        lines = [
            f"===== {system}: multiple_runs_bases.py on every range, logs in {logs_root}/<range>/<scenario> =====",
            f"{args.duration} s, bases in random quadrants, seeds from {args.seed}",
            "Mean +- std deviation over the runs of each range and scenario. The time is the one of the range, "
            "over every system and scenario",
            "",
            f"{'range':>7}  {'scenario':<24}  {'result':<12} {'time':>9}  {'runs':>4}  {'mean uncertainty':>18}  "
            f"{'above Nc (%)':>14}  {'dead at the end':>15}",
        ]
        for transmission_range, result, elapsed in summary:
            for scenario in scenarios:
                row = (f"{transmission_range:>5g} m  {scenario:<24}  {result:<12} {elapsed / 60:>5.1f} min  ")
                statistics = directory_statistics(
                    scenario_directory(system, args.logs_root, transmission_range, scenario))
                if statistics is None:
                    row += f"{'no runs':>4}"
                else:
                    runs, values = statistics
                    row += (f"{runs:>4}  {mean_std(values['mean_uncertainty']):>18}  "
                            f"{mean_std(values['above_nc']):>14}  {mean_std(values['dead'], 2):>15}")
                lines.append(row)
        lines.append("")
        lines.append(f"Total: {total:.1f} min")
        text = "\n".join(lines)
        os.makedirs(logs_root, exist_ok=True)
        with open(os.path.join(logs_root, RANGES_SUMMARY_FILE), "w") as f:
            f.write(text + "\n")
        texts.append((text, os.path.join(logs_root, RANGES_SUMMARY_FILE)))

    for text, path in texts:
        print("\n" + text)
        print(f"\nWritten to {path}")
    print(f"\nPlot them with: python3 Analytical/expanding_testing/plot_multiple_runs_bases.py "
          f"--range {' '.join(f'{r:g}' for r, _, _ in summary)}" +
          (f" --logs-root {args.logs_root}" if args.logs_root != DEFAULT_LOGS_ROOT else "") +
          " --save <name>.pdf --no-show")
    return 0 if all(result == "ok" for _, result, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
