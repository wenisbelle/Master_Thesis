"""
Runs one system many times with the same individual, each run with its recharge
base(s) somewhere else, and keeps the swarm level series of every run, so
plot_multiple_runs.py can aggregate them.

    python3 Analytical/multiple_runs.py --system egocentric --runs 20
    python3 Analytical/multiple_runs.py --system swarm --runs 10 --drones 5 --range 500
    python3 Analytical/multiple_runs.py --system expanding_testing/swarm --runs 10 --bases 2
    python3 Analytical/multiple_runs.py --system expanding_testing/egocentric --runs 10 --bases 3 --map-size 150 \
        --logs-directory multiple_runs/3_bases_150

It can be started from any directory, with gradysim on the PYTHONPATH (it is in the
container). The runs are written to <system>/multiple_runs/<range>/<drones>, the same
layout as the test_logs of all_tests_distance.py:
    run_<k>.pkl    the global uncertainty, the drones charging and the dead drones at every
                   --sample-interval, plus the base(s), the seed, the cost and the final
                   state of every drone
    config.json    the scenario shared by every run of the directory
    summary.txt    one line per run and the mean and std deviation over them, also printed

Run k is seeded with --seed + k. The seed fixes the wind and the initial destinations
of the drones, and the base(s) are drawn from it with the distribution the GA tuned the
individuals on (main.sample_base_position). So run k has the same base(s) and the same
wind on every system, swarm and egocentric are compared on the same scenarios.

The individual is the one the GA tuned for --range, the "Parameters:" line of
<system>/ga_logs/ga_logbook_<range>.txt, as in all_tests_distance.py (expanding_testing
uses the logbooks of the system it extends). Without a logbook for the range, or
without the Parameters line in it, it is an error and nothing is run. --individual
overrides it.

The runs go in parallel, a fresh process for each one, in train mode: the per-routine
protocol log of a test run is tens of MB and would be written by every process at once.
A run whose file already exists has finished before and is skipped, so --runs 20 after
--runs 10 only runs the 10 new ones. --overwrite runs them all again.
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
import importlib
import traceback
import multiprocessing

import numpy as np

##### The individuals tuned for each range, found the same way as the tests do #####
from all_tests_distance import tuned_ranges, tuned_individuals

SYSTEMS = ("swarm", "egocentric", "expanding_testing/swarm", "expanding_testing/egocentric")

ANALYTICAL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(ANALYTICAL_DIR)
##### The systems are imported as packages, Analytical.<system>.main, so src has to be on the path #####
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

##### Defaults of main_test.py and main.py, the scenario the single tests run #####
NUMBER_OF_RUNS = 10
NUMBER_OF_DRONES = 10
SIMULATION_DURATION = 10000
TRANSMISSION_RANGE = 200
GLOBAL_MAP_SAMPLE_INTERVAL = 1.0
DISTANCE_BETWEEN_CELLS = 20

##### Spread of the bases the GA tuned on, main.BASE_POSITION_SIGMA on its 100 cell map: 3 sigma #####
##### reaches the border. It is scaled with the map, whose size expanding_testing can change.   #####
BASE_POSITION_SIGMA = 333.0
BASE_POSITION_SIGMA_MAP_SIZE = 100

DEFAULT_LOGS_DIRECTORY = "multiple_runs"
CONFIG_FILE = "config.json"
SUMMARY_FILE = "summary.txt"
RUN_PATTERN = re.compile(r"run_(\d+)\.pkl$")
##### Series of GlobalMapMonitor kept on every sample, under the names the runs store them #####
SERIES = {"global_uncertainty": "uncertainty", "charging": "charging_drones", "dead": "dead_drones"}


def system_dir(system: str) -> str:
    return os.path.join(ANALYTICAL_DIR, *system.split("/"))


def system_module(system: str):
    return importlib.import_module("Analytical." + system.replace("/", ".") + ".main")


def is_expanding(system: str) -> bool:
    return system.startswith("expanding_testing/")


def tuned_system(system: str) -> str:
    """System whose GA logbooks hold the individuals, expanding_testing has no GA of its own."""
    return system.split("/")[-1]


def default_logs_directory(transmission_range: float, number_of_drones: int) -> str:
    return os.path.join(DEFAULT_LOGS_DIRECTORY, f"{transmission_range:g}", str(number_of_drones))


def run_file(run: int) -> str:
    return f"run_{run:03d}.pkl"


def sample_base_positions(seed: int, number_of_bases: int, map_size: int) -> list:
    """
    Bases of the run seeded by seed, (x, y) in metres. Each one is drawn as in
    main.sample_base_position: a Gaussian centred on the map, snapped to the nearest
    cell centre, drawn again when it falls outside the map. The bases are drawn one
    after the other from the same stream, one landing on a cell already taken is
    drawn again. With one base on the 100 cell map it is main.sample_base_position(seed).
    """
    rng = np.random.default_rng(seed)
    sigma = BASE_POSITION_SIGMA * map_size / BASE_POSITION_SIGMA_MAP_SIZE
    map_center_offset = map_size * DISTANCE_BETWEEN_CELLS / 2
    cells = []
    while len(cells) < number_of_bases:
        cell = tuple(np.round((rng.normal(0.0, sigma, size=2) + map_center_offset) / DISTANCE_BETWEEN_CELLS))
        if all(0 <= c < map_size for c in cell) and cell not in cells:
            cells.append(cell)
    return [tuple(float(c) * DISTANCE_BETWEEN_CELLS - map_center_offset for c in cell) for cell in cells]


def run_simulation(job: dict) -> dict:
    """
    One run, in a process of its own (see run_jobs). It returns what is written to
    the run file, or the traceback when the simulation crashed, so the other runs go on.
    """
    try:
        main = system_module(job["system"])
        start = time.time()
        if is_expanding(job["system"]):
            results, global_map = main.create_and_run_simulation(
                job["individual"], mode="train", sample_interval=job["sample_interval"],
                number_of_drones=job["drones"], seed=job["seed"],
                charging_base_positions=job["base_positions"],
                charging_capacity=job["charging_capacity"])
        else:
            results, global_map = main.create_and_run_simulation(
                job["individual"], mode="train", sample_interval=job["sample_interval"],
                number_of_drones=job["drones"], seed=job["seed"],
                charging_base_position=job["base_positions"][0])

        record = {
            "run": job["run"],
            "seed": job["seed"],
            "base_positions": job["base_positions"],
            "nc_charging_base": global_map.nc_charging_base,
            "times": np.array(global_map.times),
            **{name: np.array(getattr(global_map, attribute)) for name, attribute in SERIES.items()},
            "cost": float(main.evaluate_simulation_cost(results, global_map)),
            "cost_terms": main.simulation_cost_terms(global_map),
            "drones": results,
            "elapsed": time.time() - start,
        }
        ##### Only expanding_testing splits the drones charging by base #####
        if hasattr(global_map, "charging_drones_per_base"):
            record["charging_per_base"] = np.array(global_map.charging_drones_per_base)
        return record
    except Exception:
        return {"run": job["run"], "seed": job["seed"], "error": traceback.format_exc()}


def time_grid(config: dict) -> np.ndarray:
    """
    Instants every run is read at, one per sample_interval. The monitor samples on
    the same grid, but its first sample is the first event and the last one is
    repeated on the event that ends the run, so the runs are put back on it.
    """
    return np.arange(0.0, config["duration"], config["sample_interval"])


def aligned(record: dict, grid: np.ndarray) -> dict:
    """The series of the run on grid: the value at t is the last one sampled at or before t."""
    index = np.clip(np.searchsorted(record["times"], grid, side="right") - 1, 0, None)
    return {name: np.asarray(record[name])[index] for name in SERIES}


def run_statistics(record: dict, grid: np.ndarray) -> dict:
    series = aligned(record, grid)
    return {
        "mean_uncertainty": float(series["global_uncertainty"].mean()),
        ##### Share of the run with more than Nc drones charging at once, in % #####
        "above_nc": 100.0 * float(np.mean(series["charging"] > record["nc_charging_base"])),
        "max_charging": int(series["charging"].max()),
        "dead": int(series["dead"][-1]),
        "cost": record["cost"],
    }


def load_runs(directory: str):
    """The config of the directory and its runs, sorted by run number."""
    with open(os.path.join(directory, CONFIG_FILE)) as f:
        config = json.load(f)
    runs = []
    for name in os.listdir(directory):
        if RUN_PATTERN.match(name):
            with open(os.path.join(directory, name), "rb") as f:
                runs.append(pickle.load(f))
    return config, sorted(runs, key=lambda record: record["run"])


def format_position(position) -> str:
    return "(" + ", ".join(f"{c:g}" for c in position) + ")"


def summary(config: dict, runs: list) -> str:
    grid = time_grid(config)
    statistics = [run_statistics(record, grid) for record in runs]
    nc = sorted({record["nc_charging_base"] for record in runs})
    lines = [
        f"{config['system']}: {len(runs)} runs, {config['drones']} drones, {config['transmission_range']:g} m, "
        f"{config['duration']} s, {config['bases']} base(s) on a {config['map_size']}x{config['map_size']} map, "
        f"Nc = {', '.join(str(n) for n in nc)}",
        f"Individual: {config['individual']}",
        "",
        f"{'run':>4} {'seed':>6} {'mean unc':>10} {'above Nc':>9} {'max charging':>13} {'dead':>5} "
        f"{'cost':>10}  base(s)",
    ]
    for record, stats in zip(runs, statistics):
        lines.append(f"{record['run']:>4} {record['seed']:>6} {stats['mean_uncertainty']:>10.1f} "
                     f"{stats['above_nc']:>8.1f}% {stats['max_charging']:>13} {stats['dead']:>5} "
                     f"{stats['cost']:>10.1f}  "
                     + " ".join(format_position(position) for position in record["base_positions"]))

    ##### Over the runs, a single run has no spread #####
    def spread(name):
        values = np.array([stats[name] for stats in statistics], dtype=float)
        return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0

    lines.append("")
    lines.append("Mean +- std deviation over the runs:")
    lines.append("  mean global uncertainty: {:.1f} +- {:.1f}".format(*spread("mean_uncertainty")))
    lines.append("  time with more than Nc drones charging: {:.1f} +- {:.1f} %".format(*spread("above_nc")))
    lines.append("  most drones charging at once: {:.2f} +- {:.2f}".format(*spread("max_charging")))
    lines.append("  dead drones at the end: {:.2f} +- {:.2f}".format(*spread("dead")))
    lines.append("  cost: {:.1f} +- {:.1f}".format(*spread("cost")))
    return "\n".join(lines)


def run_jobs(jobs: list, workers: int, logs_directory: str, config: dict) -> int:
    """Runs the jobs in parallel and writes each run as soon as it finishes. Returns how many failed."""
    failed = 0
    grid = time_grid(config)
    ##### Forked: the workers have to inherit the module constants main() changed, as in the GA. #####
    ##### One process per run, so nothing a run leaves in a module can reach the next one.        #####
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=workers, maxtasksperchild=1) as pool:
        for done, record in enumerate(pool.imap_unordered(run_simulation, jobs), start=1):
            if "error" in record:
                failed += 1
                print(f"[{done}/{len(jobs)}] run {record['run']} (seed {record['seed']}) failed:\n"
                      f"{record['error']}", flush=True)
                continue
            path = os.path.join(logs_directory, run_file(record["run"]))
            with open(path, "wb") as f:
                pickle.dump(record, f)
            stats = run_statistics(record, grid)
            print(f"[{done}/{len(jobs)}] run {record['run']} (seed {record['seed']}, base(s) "
                  + " ".join(format_position(p) for p in record["base_positions"])
                  + f"): mean global uncertainty {stats['mean_uncertainty']:.1f}, "
                  f"more than Nc = {record['nc_charging_base']} charging {stats['above_nc']:.1f}% of the time, "
                  f"{stats['dead']} dead, cost {stats['cost']:.1f}, {record['elapsed'] / 60:.1f} min -> {path}",
                  flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Many runs of one system, each with its recharge base(s) somewhere else")
    parser.add_argument("--system", choices=SYSTEMS, required=True,
                        help="which system to run")
    parser.add_argument("--runs", type=int, default=NUMBER_OF_RUNS,
                        help="number of runs (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=NUMBER_OF_DRONES,
                        help="number of drones (default: %(default)s)")
    parser.add_argument("--range", type=float, default=TRANSMISSION_RANGE, dest="transmission_range",
                        help="communication range of the drones, in m. Also picks the tuned individual "
                             "(default: %(default)s)")
    parser.add_argument("--duration", type=int, default=SIMULATION_DURATION,
                        help="simulated seconds of each run (default: %(default)s)")
    parser.add_argument("--sample-interval", type=float, default=GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map, the "
                             "integration step of the cost (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="run k is seeded with SEED + k, which also draws its base(s) (default: %(default)s)")
    parser.add_argument("--individual", type=float, nargs="+",
                        help="genes of the individual, in the GENE_BOUNDS order of the system's main.py "
                             "(default: the one tuned for --range, an error when there is none)")
    parser.add_argument("--bases", type=int, default=1,
                        help="expanding_testing only: recharge bases of each run, all drawn again on "
                             "every run (default: %(default)s)")
    parser.add_argument("--map-size", type=int, metavar="CELLS",
                        help=f"expanding_testing only: side of the square map, in cells {DISTANCE_BETWEEN_CELLS} m "
                             "apart (default: MAP_WIDTH of main.py)")
    parser.add_argument("--charging-capacity", type=int, metavar="NC",
                        help="expanding_testing only: most drones charging at once over all the bases "
                             "(default: int(drones / 3))")
    parser.add_argument("--logs-directory", "--logs_directory",
                        help="directory, relative to the system directory, the runs are written to "
                             "(default: multiple_runs/<range>/<drones>)")
    parser.add_argument("--workers", type=int,
                        help="runs simulated at once (default: one per run, at most one per CPU)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the runs already in the directory and run them all again")
    args = parser.parse_args()

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

    main_module = system_module(args.system)

    ##### Module constants, read when each simulation is built. The workers are forked #####
    ##### after this point, so they get these values too.                               #####
    main_module.SIMULATION_DURATION = args.duration
    main_module.TRANSMISSION_RANGE = args.transmission_range
    if args.map_size is not None:
        ##### The protocol shares one centre offset between both axes, the map has to be square #####
        main_module.MAP_WIDTH = main_module.MAP_HEIGHT = args.map_size
    map_size = main_module.MAP_WIDTH

    if args.individual is not None:
        individual, origin = list(args.individual), "--individual"
    else:
        ##### No fallback to BEST_INDIVIDUAL of main.py, the range would run untuned #####
        try:
            individual = tuned_individuals(tuned_system(args.system), [args.transmission_range])[
                args.transmission_range]
        except ValueError as error:
            parser.error(f"no tuned individual, nothing was run:\n{error}\n"
                         f"Tune the range first, or give the individual with --individual")
        origin = tuned_ranges(tuned_system(args.system))[args.transmission_range]
    if len(individual) != len(main_module.GENE_BOUNDS):
        parser.error(f"the individual has {len(individual)} genes, {args.system} takes "
                     f"{len(main_module.GENE_BOUNDS)}: {[name for name, _, _ in main_module.GENE_BOUNDS]}")

    logs_directory = os.path.join(system_dir(args.system), args.logs_directory or
                                  default_logs_directory(args.transmission_range, args.drones))
    config = {
        "system": args.system,
        "drones": args.drones,
        "transmission_range": args.transmission_range,
        "duration": args.duration,
        "sample_interval": args.sample_interval,
        "seed": args.seed,
        "individual": individual,
        "map_size": map_size,
        "bases": args.bases,
        "charging_capacity": args.charging_capacity,
    }

    ##### The runs of a directory are aggregated together, so they all have to come from #####
    ##### the same scenario. Another scenario goes to another directory, or replaces them. #####
    config_path = os.path.join(logs_directory, CONFIG_FILE)
    os.makedirs(logs_directory, exist_ok=True)
    existing = [name for name in os.listdir(logs_directory) if RUN_PATTERN.match(name)]
    if args.overwrite:
        for name in existing + [CONFIG_FILE, SUMMARY_FILE]:
            if os.path.exists(os.path.join(logs_directory, name)):
                os.remove(os.path.join(logs_directory, name))
        existing = []
    elif os.path.exists(config_path):
        with open(config_path) as f:
            previous = json.load(f)
        ##### Compared as JSON, the way it was read back #####
        current = json.loads(json.dumps(config))
        different = [key for key in current if previous.get(key) != current[key]]
        if different and existing:
            parser.error(f"{logs_directory} holds {len(existing)} runs of another scenario "
                         f"(different {', '.join(different)}). Give another --logs-directory, "
                         f"or --overwrite to replace them")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    jobs = []
    for run in range(args.runs):
        if os.path.exists(os.path.join(logs_directory, run_file(run))):
            continue
        seed = args.seed + run
        jobs.append({"system": args.system, "run": run, "seed": seed, "individual": individual,
                     "drones": args.drones, "sample_interval": args.sample_interval,
                     "charging_capacity": args.charging_capacity,
                     "base_positions": sample_base_positions(seed, args.bases, map_size)})

    print(f"{args.system}: {args.runs} runs of {args.drones} drones, {args.transmission_range:g} m, "
          f"{args.duration} s, {args.bases} base(s) drawn again on every run")
    print(f"Individual ({origin}): {individual}")
    print(f"Logs: {logs_directory}")
    skipped = args.runs - len(jobs)
    if skipped:
        print(f"{skipped} run(s) already in the directory, skipped")

    failed = 0
    if jobs:
        workers = args.workers or min(len(jobs), os.cpu_count() or 1)
        print(f"Running {len(jobs)} run(s), {workers} at once ...", flush=True)
        failed = run_jobs(jobs, workers, logs_directory, config)

    config, runs = load_runs(logs_directory)
    if runs:
        text = summary(config, runs)
        with open(os.path.join(logs_directory, SUMMARY_FILE), "w") as f:
            f.write(text + "\n")
        print()
        print(text)
        print(f"\nPlot them with: python3 Analytical/plot_multiple_runs.py --system {args.system} "
              + (f"--logs-directory {args.logs_directory}" if args.logs_directory else
                 f"--range {args.transmission_range:g} --drones {args.drones}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
