"""
Runs the GA tuning (main.py --mode train) once for every TRANSMISSION_RANGE, one
after the other, so the effect of the communication range on the system can be
compared.

    python3 Analytical/all_optimizations_distance.py                      # swarm, every range
    python3 Analytical/all_optimizations_distance.py --system egocentric
    python3 Analytical/all_optimizations_distance.py --ranges 100 200     # only these ranges

It can be started from any directory. Each range writes, in the directory of the
system (swarm/ or egocentric/):
    ga_logbook_<range>.txt   the logbook of main.py, written when its GA finishes
    ga_output_<range>.txt    everything main.py printed, also shown here, so a run
                             that is interrupted still leaves its generations behind

Every range is a separate process, the same as running main.py by hand, so one
run cannot leak state into the next. A range whose logbook already exists has
finished before and is skipped, a sweep that was interrupted can be started again
without repeating it. --overwrite runs it again.
"""

import os
import sys
import time
import argparse
import subprocess

TRANSMISSION_RANGES = (100, 200, 500, 750, 1000, 2000)  # m
SYSTEMS = ("swarm", "egocentric")

ANALYTICAL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(ANALYTICAL_DIR)


def run_range(system: str, transmission_range: float, logbook: str, output: str) -> int:
    """Runs one GA tuning and returns its exit status."""
    command = [sys.executable, "-u", "-m", f"Analytical.{system}.main", "--mode", "train",
               "--transmission-range", f"{transmission_range:g}", "--logbook", logbook]

    ##### main.py is imported as a package, so src has to be on the path #####
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (SRC_DIR, env.get("PYTHONPATH")) if p)

    with open(output, "w") as out:
        process = subprocess.Popen(command, cwd=SRC_DIR, env=env, text=True, bufsize=1,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            sys.stdout.write(line)
            out.write(line)
            out.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="GA tuning for every transmission range, sequentially")
    parser.add_argument("--system", choices=SYSTEMS, default="swarm",
                        help="which system to tune (default: %(default)s)")
    parser.add_argument("--ranges", type=float, nargs="+", default=TRANSMISSION_RANGES,
                        help="transmission ranges to tune for, in m (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="also run the ranges whose logbook already exists")
    args = parser.parse_args()

    system_dir = os.path.join(ANALYTICAL_DIR, args.system)
    summary = []
    for transmission_range in args.ranges:
        logbook = os.path.join(system_dir, f"ga_logbook_{transmission_range:g}.txt")
        output = os.path.join(system_dir, f"ga_output_{transmission_range:g}.txt")

        if os.path.exists(logbook) and not args.overwrite:
            print(f"##### {args.system}, {transmission_range:g} m: {logbook} already exists, skipped #####")
            summary.append((transmission_range, "skipped", 0.0, logbook))
            continue

        print(f"##### {args.system}, {transmission_range:g} m: tuning, logbook {logbook} #####", flush=True)
        start = time.time()
        try:
            status = run_range(args.system, transmission_range, logbook, output)
        except KeyboardInterrupt:
            print(f"##### Interrupted during {transmission_range:g} m, the remaining ranges were not run #####")
            summary.append((transmission_range, "interrupted", time.time() - start, output))
            break
        summary.append((transmission_range, "ok" if status == 0 else f"failed ({status})",
                        time.time() - start, logbook if status == 0 else output))

    print(f"\n===== {args.system}: transmission range sweep =====")
    for transmission_range, result, elapsed, path in summary:
        print(f"{transmission_range:>7g} m  {result:<12} {elapsed / 3600:7.1f} h  {path}")
    return 0 if all(result in ("ok", "skipped") for _, result, _, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
