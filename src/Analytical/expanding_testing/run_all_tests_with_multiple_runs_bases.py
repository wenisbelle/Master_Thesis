"""
Runs multiple_runs_bases.py for every TRANSMISSION_RANGE the GA tuned, one range after
the other, so the systems are compared on a multi-base scenario over many runs, each
with its bases in random quadrants. It is
Analytical/run_all_tests_with_multiple_runs.py for expanding_testing.

    cd src
    export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm egocentric naive
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm --ranges 100 200
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system egocentric naive \
        --cells 200 --drones 20 --bases 3 --runs 40
    python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm egocentric naive \
        --hand-tuned

It can be started from any directory. The ranges are the ones tuned by
all_optimizations_distance.py, found in Analytical/<system>/ga_logs/ga_logbook_<range>.txt
(egocentric's for naive), for every system given. Each system runs with the individual
multiple_runs_bases.py gives it for the range (--hand-tuned: HAND_TUNED_INDIVIDUAL of
Analytical/<system>/main_test.py on every range). A range without one for one of the
systems is an error: nothing is run.

The scenario is the one of multiple_runs_bases.py, --cells, --drones and --bases, the
same on every range. Every range runs it on every system in one call to
multiple_runs_bases.py, so all of its runs share one pool of processes. They are
written to <system>/<logs-root>/<range>/<scenario>, here in expanding_testing.

Run k of every range is seeded with --seed + k, so every range runs the same bases and
the same wind. The runs already in a directory are skipped by multiple_runs_bases.py,
so a sweep that was interrupted can be started again without repeating them, and
--runs 40 after --runs 20 only runs the new ones. At the end the mean +- std deviation
over the runs of every range is printed and written to
<system>/<logs-root>/ranges_summary_<scenario>.txt.
"""

import os
import sys
import time
import argparse

EXPANDING_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICAL_DIR = os.path.dirname(EXPANDING_DIR)
if ANALYTICAL_DIR not in sys.path:
    sys.path.insert(0, ANALYTICAL_DIR)

from all_tests_distance import tuned_ranges, ga_logs_dir
from multiple_runs import SIMULATION_DURATION, GLOBAL_MAP_SAMPLE_INTERVAL, system_dir, tuned_system
from run_all_tests_with_multiple_runs import RANGES_SUMMARY_FILE, run_range, mean_std, directory_statistics
from multiple_runs_bases import (SYSTEMS, NUMBER_OF_RUNS, DEFAULT_LOGS_ROOT, HAND_TUNED_LOGS_ROOT,
                                 scenario_directory, scenario_name, add_scenario_arguments,
                                 check_scenario_arguments, system_individual)

MULTIPLE_RUNS_BASES = os.path.join(EXPANDING_DIR, "multiple_runs_bases.py")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="multiple_runs_bases.py for every tuned transmission range, sequentially")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", required=True,
                        help="system(s) to run, from expanding_testing")
    parser.add_argument("--ranges", type=float, nargs="+",
                        help="transmission ranges to run, in m (default: every range in "
                             "Analytical/<system>/ga_logs, of every system given)")
    add_scenario_arguments(parser)
    parser.add_argument("--hand-tuned", "--hand_tuned", action="store_true",
                        help="run HAND_TUNED_INDIVIDUAL of Analytical/<system>/main_test.py on every range, "
                             "instead of the tuned one")
    parser.add_argument("--logs-root",
                        help="directory, relative to each system directory, the runs are written to, "
                             f"as <range>/<scenario> (default: {DEFAULT_LOGS_ROOT}, "
                             f"{HAND_TUNED_LOGS_ROOT} with --hand-tuned)")

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
    check_scenario_arguments(parser, args)
    if args.logs_root is None:
        args.logs_root = HAND_TUNED_LOGS_ROOT if args.hand_tuned else DEFAULT_LOGS_ROOT
    systems = [SYSTEMS[name] for name in dict.fromkeys(args.system)]
    scenario = scenario_name(args.cells, args.drones, args.bases)

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
        for transmission_range in ranges:
            try:
                system_individual(system, transmission_range, args.hand_tuned)
            except ValueError as error:
                errors.append(f"{system}, {transmission_range:g} m: {error}")
    if errors:
        parser.error("no individual, nothing was run:\n" + "\n".join(errors))

    summary = []
    sweep_start = time.time()
    for done, transmission_range in enumerate(ranges, start=1):
        command = [sys.executable, "-u", MULTIPLE_RUNS_BASES,
                   "--system", *args.system,
                   "--cells", str(args.cells),
                   "--drones", str(args.drones),
                   "--bases", str(args.bases),
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
        if args.hand_tuned:
            command.append("--hand-tuned")

        print(f"\n##### [{done}/{len(ranges)}] {transmission_range:g} m: {', '.join(args.system)}, "
              f"{scenario}, {args.runs} runs each #####", flush=True)
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
            f"{scenario}, {args.duration} s, bases in random quadrants, seeds from {args.seed}"
            + (", HAND_TUNED_INDIVIDUAL" if args.hand_tuned else ""),
            "Mean +- std deviation over the runs of each range. The time is the one of the range, "
            "over every system",
            "",
            f"{'range':>7}  {'scenario':<24}  {'result':<12} {'time':>9}  {'runs':>4}  {'mean uncertainty':>18}  "
            f"{'above Nc (%)':>14}  {'dead at the end':>15}",
        ]
        for transmission_range, result, elapsed in summary:
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
        ##### One per scenario, a sweep of another scenario does not overwrite it #####
        stem, extension = os.path.splitext(RANGES_SUMMARY_FILE)
        summary_path = os.path.join(logs_root, f"{stem}_{scenario}{extension}")
        with open(summary_path, "w") as f:
            f.write(text + "\n")
        texts.append((text, summary_path))

    for text, path in texts:
        print("\n" + text)
        print(f"\nWritten to {path}")
    print(f"\nPlot them with: python3 Analytical/expanding_testing/plot_multiple_runs_bases.py "
          f"--range {' '.join(f'{r:g}' for r, _, _ in summary)} "
          f"--cells {args.cells} --drones {args.drones} --bases {args.bases}" +
          (" --hand-tuned" if args.hand_tuned else "") +
          (f" --logs-root {args.logs_root}"
           if args.logs_root != (HAND_TUNED_LOGS_ROOT if args.hand_tuned else DEFAULT_LOGS_ROOT) else "") +
          " --save <name>.pdf --no-show")
    return 0 if all(result == "ok" for _, result, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
