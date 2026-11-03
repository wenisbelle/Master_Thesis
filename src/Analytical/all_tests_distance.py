"""
Runs the test simulation (swarm/main_test.py) with the individual the GA tuned for
every TRANSMISSION_RANGE, on every swarm size, one after the other, so the tuned
systems can be compared.

    python3 Analytical/all_tests_distance.py                          # swarm, every range, 3/5/7/10 drones
    python3 Analytical/all_tests_distance.py --system egocentric
    python3 Analytical/all_tests_distance.py --ranges 100 200 --drones 5 10
    python3 Analytical/all_tests_distance.py --base-position 400 -200 --test-logs test_logs_base_400_-200

It can be started from any directory. The ranges are the ones tuned by
all_optimizations_distance.py, found in <system>/ga_logs/ga_logbook_<range>.txt. The
individual of a range is the best of the last generation, the "Parameters:" line
of its logbook (main.py explains why it is preferred over the Hall of Fame). A range
without a logbook, or without the Parameters line in it, is an error: nothing is run.

Each range and swarm size writes, in <system>/test_logs/<range>/<drones>:
    main_test.log        the protocol log
    test_recording.pkl   the recording, plot it from swarm/plots with
                         python3 plot_drones.py --directory test_logs/<range>/<drones>
                         from <system>/plots
    test_output.txt      the report main_test.py printed, also shown here

The recharge base is the one of main_test.py, the map centre, unless --base-position
is given. Runs with another base should go to their own --test-logs directory,
otherwise the runs already done at the centre are found there and skipped.

Every run is a separate process, the same as running main_test.py by hand. A run
whose recording already exists has finished before and is skipped, a sweep that was
interrupted can be started again without repeating it. --overwrite runs it again.
"""

import os
import re
import ast
import sys
import time
import argparse
import threading
import subprocess
from collections import deque

DRONE_COUNTS = (3, 5, 7, 10)
SYSTEMS = ("swarm", "egocentric")

ANALYTICAL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(ANALYTICAL_DIR)
LOGBOOK_PATTERN = re.compile(r"ga_logbook_(\d+(?:\.\d+)?)\.txt$")

##### Names main_test.py gives its files inside the logs directory #####
RECORDING_FILE = "test_recording.pkl"
OUTPUT_FILE = "test_output.txt"

##### Lines of stderr kept to show why a run crashed. main_test.py also writes the #####
##### whole protocol log there (gradysim's console handler), tens of MB per run.    #####
STDERR_TAIL_LINES = 40


def system_dir(system: str) -> str:
    return os.path.join(ANALYTICAL_DIR, system)


def ga_logs_dir(system: str) -> str:
    return os.path.join(system_dir(system), "ga_logs")


def tuned_ranges(system: str) -> dict:
    """Maps each range tuned in the ga_logs directory of the system to its logbook."""
    ranges = {}
    if not os.path.isdir(ga_logs_dir(system)):
        return ranges
    for name in os.listdir(ga_logs_dir(system)):
        match = LOGBOOK_PATTERN.match(name)
        if match:
            ranges[float(match.group(1))] = os.path.join(ga_logs_dir(system), name)
    return dict(sorted(ranges.items()))


def best_individual(logbook: str) -> list:
    """
    The best individual of the last generation, written by main.py at the end of the
    logbook. Raises ValueError when the logbook has none.
    """
    with open(logbook) as f:
        parameters = [line for line in f if line.startswith("Parameters:")]
    if not parameters:
        raise ValueError(f"{logbook} has no Parameters line, the tuning did not finish")
    try:
        return [float(value) for value in ast.literal_eval(parameters[-1].split(":", 1)[1].strip())]
    except (ValueError, SyntaxError, TypeError) as error:
        raise ValueError(f"{logbook}: its Parameters line is not a list of numbers ({error})") from None


def tuned_individuals(system: str, ranges) -> dict:
    """
    Maps each range to the individual tuned for it in the ga_logs of the system. Raises
    ValueError, naming every range at fault, when one has no logbook or no individual
    in it. No range falls back to another individual, it would be tested untuned.
    """
    logbooks = tuned_ranges(system)
    individuals, errors = {}, []
    for transmission_range in ranges:
        if transmission_range not in logbooks:
            errors.append(f"{transmission_range:g} m: no ga_logbook_{transmission_range:g}.txt "
                          f"in {ga_logs_dir(system)}")
            continue
        try:
            individuals[transmission_range] = best_individual(logbooks[transmission_range])
        except ValueError as error:
            errors.append(f"{transmission_range:g} m: {error}")
    if errors:
        raise ValueError("\n".join(errors))
    return individuals


def recorded_since(logs_directory: str, start: float) -> bool:
    recording = os.path.join(logs_directory, RECORDING_FILE)
    return os.path.exists(recording) and os.path.getmtime(recording) >= start


def run_test(system: str, transmission_range: float, number_of_drones: int, individual: list,
             base_position, logs_directory: str) -> str:
    """
    Runs one main_test.py of the system, writing into logs_directory (relative to
    the system directory), and returns how it ended.
    """
    absolute_logs_directory = os.path.join(system_dir(system), logs_directory)
    os.makedirs(absolute_logs_directory, exist_ok=True)
    command = [sys.executable, "-u", "-m", f"Analytical.{system}.main_test",
               "--drones", str(number_of_drones),
               "--transmission-range", f"{transmission_range:g}",
               "--logs_directory", logs_directory,
               "--individual", *[repr(value) for value in individual]]
    if base_position is not None:
        command += ["--base-position", *[f"{c:g}" for c in base_position]]

    ##### main_test.py is imported as a package, so src has to be on the path #####
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC_DIR, env.get("PYTHONPATH")) if p)

    start = time.time()
    with open(os.path.join(absolute_logs_directory, OUTPUT_FILE), "w") as out:
        process = subprocess.Popen(command, cwd=SRC_DIR, env=env, text=True, bufsize=1,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr_tail = deque(maxlen=STDERR_TAIL_LINES)
        stderr_reader = threading.Thread(target=stderr_tail.extend, args=(process.stderr,), daemon=True)
        stderr_reader.start()
        for line in process.stdout:
            sys.stdout.write(line)
            out.write(line)
            out.flush()
        status = process.wait()
        stderr_reader.join()

        if status == 0:
            return "ok"
        ##### main_test.py also exits with 1 when a check fails, after saving the recording. #####
        ##### Without a fresh recording it crashed, and the traceback is in the tail.        #####
        if recorded_since(absolute_logs_directory, start):
            return "checks failed"
        tail = f"----- last lines of stderr -----\n{''.join(stderr_tail)}"
        sys.stdout.write(tail)
        out.write(tail)
        return f"failed ({status})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="main_test.py for every tuned transmission range and swarm size, sequentially")
    parser.add_argument("--system", choices=SYSTEMS, default="swarm",
                        help="which system to test (default: %(default)s)")
    parser.add_argument("--ranges", type=float, nargs="+",
                        help="transmission ranges to test, in m (default: every range in "
                             "<system>/ga_logs)")
    parser.add_argument("--drones", type=int, nargs="+", default=DRONE_COUNTS,
                        help="swarm sizes to test (default: %(default)s)")
    parser.add_argument("--base-position", type=float, nargs=2, metavar=("X", "Y"),
                        help="position of the recharge base, in m, the map centre is (0, 0) "
                             "(default: the one of main_test.py, the map centre)")
    parser.add_argument("--test-logs", default="test_logs",
                        help="directory, relative to the system directory, the runs are written to, "
                             "as <range>/<drones> (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="also run the ones whose recording already exists")
    args = parser.parse_args()

    ranges = args.ranges if args.ranges is not None else list(tuned_ranges(args.system))
    if not ranges:
        parser.error(f"no ga_logbook_<range>.txt in {ga_logs_dir(args.system)}, nothing to test")
    ##### Every individual is read before the first run, so a range without one stops #####
    ##### the sweep here and not after hours of the others.                             #####
    try:
        individuals = tuned_individuals(args.system, ranges)
    except ValueError as error:
        parser.error(f"no tuned individual, nothing was run:\n{error}")
    base = "the map centre" if args.base_position is None else tuple(args.base_position)

    summary = []
    interrupted = False
    for transmission_range in ranges:
        individual = individuals[transmission_range]
        for number_of_drones in args.drones:
            logs_directory = os.path.join(args.test_logs, f"{transmission_range:g}", str(number_of_drones))
            absolute_logs_directory = os.path.join(system_dir(args.system), logs_directory)
            recording = os.path.join(absolute_logs_directory, RECORDING_FILE)

            if os.path.exists(recording) and not args.overwrite:
                print(f"##### {transmission_range:g} m, {number_of_drones} drones: "
                      f"{recording} already exists, skipped #####")
                summary.append((transmission_range, number_of_drones, "skipped", 0.0, absolute_logs_directory))
                continue

            print(f"##### {transmission_range:g} m, {number_of_drones} drones, base at {base}: "
                  f"testing {individual}, logs {absolute_logs_directory} #####", flush=True)
            start = time.time()
            try:
                result = run_test(args.system, transmission_range, number_of_drones, individual,
                                  args.base_position, logs_directory)
            except KeyboardInterrupt:
                print(f"##### Interrupted during {transmission_range:g} m, {number_of_drones} drones, "
                      f"the remaining runs were not run #####")
                summary.append((transmission_range, number_of_drones, "interrupted",
                                time.time() - start, absolute_logs_directory))
                interrupted = True
                break
            summary.append((transmission_range, number_of_drones, result,
                            time.time() - start, absolute_logs_directory))
        if interrupted:
            break

    print(f"\n===== {args.system}: tests of the tuned individuals, base at {base} =====")
    for transmission_range, number_of_drones, result, elapsed, path in summary:
        drones = f"{number_of_drones} drones"
        print(f"{transmission_range:>7g} m  {drones:<10} {result:<14} {elapsed / 60:7.1f} min  {path}")
    return 0 if all(result in ("ok", "skipped") for _, _, result, _, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
