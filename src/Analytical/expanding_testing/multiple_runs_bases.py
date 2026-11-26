"""
Runs the expanding_testing systems many times on a multi-base scenario, each run with
its recharge bases placed again, and keeps the swarm level series of every run, so
plot_multiple_runs_bases.py can aggregate them. It is Analytical/multiple_runs.py with
the scenario and the base placement of the multi-base tests.

    cd src
    export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen
    python3 Analytical/expanding_testing/multiple_runs_bases.py --system swarm egocentric naive   # 100x100, 10 drones, 2 bases, 100 m
    python3 Analytical/expanding_testing/multiple_runs_bases.py --system swarm egocentric naive \
        --cells 200 --drones 20 --bases 3 --range 500
    python3 Analytical/expanding_testing/multiple_runs_bases.py --system swarm egocentric naive --hand-tuned
    python3 Analytical/expanding_testing/multiple_runs_bases.py --system egocentric --runs 40

It can be started from any directory, with gradysim on the PYTHONPATH (it is in the
container).

The scenario is set by three flags, one run of the script is one scenario:
    --cells N    side of the square map, N x N cells 20 m apart (100 -> 2000 m x 2000 m)
    --drones N   drones of the swarm
    --bases N    recharge bases, 1 to 4 (one per quadrant, see below). --base works too
Its name is <cells>x<cells>_<drones>drones_<bases>bases, the directory the runs go to.
Nc is default_charging_capacity of main.py: CHARGING_BASE_CAPACITY (2), else int(drones / 3), over all the bases together.

Base placement. The map is split into its 4 quadrants. Each base takes a quadrant of
its own, drawn at random among the ones still free, and its position is drawn from a
Gaussian centred on the centre of that quadrant, with a sigma of a third of half the
side of the quadrant: 3 sigma reaches the border of the quadrant, the way
BASE_POSITION_SIGMA reaches the border of the map in the GA. That is 167 m on the
100 cell map and 333 m on the 200 cell one. The position is snapped to the nearest
cell centre, and drawn again when that cell is outside the quadrant. Two bases are
never in the same quadrant, they can be side by side or diagonal.

Run k is seeded with --seed + k. The seed fixes the wind and the initial destinations
of the drones, and the bases are drawn from it too, so:
- run k of every system has the same bases and the same wind, the systems are
  compared on the same scenarios;
- run k has the same bases on every range and every number of drones;
- run k with 3 bases has the 2 bases of the 2 base scenario on the same map, plus a
  third one.

The individual of each system is the one Analytical/<system> runs at --range, the
system expanding_testing/<system> extends:
- swarm and egocentric: the "Parameters:" line of
  Analytical/<system>/ga_logs/ga_logbook_<range>.txt;
- naive: naive.main.default_individual, its own logbook once it is tuned, until then
  the coverage genes of the egocentric individual tuned for the range;
- with --hand-tuned, HAND_TUNED_INDIVIDUAL of Analytical/<system>/main_test.py, the same
  for every range.
Without one for a system, it is an error and nothing is run.

The runs are written to <system>/<logs-root>/<range>/<scenario>, here in
expanding_testing, with the files of Analytical/multiple_runs.py (run_<k>.pkl,
config.json, summary.txt), so plot_multiple_runs.py reads them too. --logs-root is
multiple_runs by default, multiple_runs_hand_tuned with --hand-tuned, so the hand-tuned
runs never mix with the tuned ones. Each run also keeps the drones charging on each
base and the quadrant of each base.

The runs of every system go in one pool, a fresh process for each one, in train mode.
A run whose file already exists has finished before and is skipped, so --runs 40 after
--runs 20 only runs the 20 new ones. --overwrite runs them all again. A directory that
holds runs of another configuration (individual, duration, seed, initial uncertainty
...) is refused.
"""

import os
import re
import sys
import json
import pickle
import argparse
import traceback
import multiprocessing

import numpy as np

EXPANDING_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICAL_DIR = os.path.dirname(EXPANDING_DIR)
##### The runs, the record of a run and its statistics are the ones of Analytical/multiple_runs.py #####
if ANALYTICAL_DIR not in sys.path:
    sys.path.insert(0, ANALYTICAL_DIR)

from all_tests_distance import system_individuals
from multiple_runs import (SIMULATION_DURATION, TRANSMISSION_RANGE, GLOBAL_MAP_SAMPLE_INTERVAL,
                           DISTANCE_BETWEEN_CELLS, CONFIG_FILE, SUMMARY_FILE, RUN_PATTERN,
                           DEFAULT_LOGS_DIRECTORY, HAND_TUNED_LOGS_DIRECTORY,
                           system_dir, system_module, gene_names, run_file, run_simulation,
                           load_runs, summary, time_grid, run_statistics, format_position)
from run_all_tests_with_multiple_runs import mean_std

##### Command line name of each system, and the directory it runs from #####
SYSTEMS = {"swarm": "expanding_testing/swarm", "egocentric": "expanding_testing/egocentric",
           "naive": "expanding_testing/naive"}

##### The scenario of a run of the script: --cells, --drones, --bases #####
MAP_SIZE = 100
NUMBER_OF_DRONES = 10
NUMBER_OF_BASES = 2

NUMBER_OF_RUNS = 20
DEFAULT_LOGS_ROOT = DEFAULT_LOGS_DIRECTORY
HAND_TUNED_LOGS_ROOT = HAND_TUNED_LOGS_DIRECTORY

##### Quadrant q covers the low (0) or high (1) half of the cells on x (q % 2) and on y (q // 2) #####
QUADRANT_NAMES = ("SW", "SE", "NW", "NE")
##### Sigma of the Gaussian of a base, as a share of half the side of its quadrant: 3 sigma reaches its border #####
BASE_SIGMA_IN_QUADRANT = 1.0 / 3.0

SCENARIO_PATTERN = re.compile(r"^(\d+)x\1_(\d+)drones_(\d+)bases$")


def scenario_name(map_size: int, drones: int, bases: int) -> str:
    """Name of the scenario, the directory its runs go to: <cells>x<cells>_<drones>drones_<bases>bases."""
    return f"{map_size}x{map_size}_{drones}drones_{bases}bases"


def parse_scenario(name: str):
    """(map_size, drones, bases) of a scenario name, None when name is not one."""
    match = SCENARIO_PATTERN.match(name)
    return tuple(int(value) for value in match.groups()) if match else None


def scenario_directory(system: str, logs_root: str, transmission_range: float, scenario: str) -> str:
    return os.path.join(system_dir(system), logs_root, f"{transmission_range:g}", scenario)


def scenario_label(scenario: str) -> str:
    map_size, drones, bases = parse_scenario(scenario)
    return (f"{bases} base{'s' if bases != 1 else ''} in random quadrants, {map_size}x{map_size} map, "
            f"{drones} drones")


def source_system(system: str) -> str:
    """The system of Analytical that expanding_testing/<system> extends, and whose individual it runs."""
    return system.split("/")[-1]


def base_sigma(map_size: int) -> float:
    """Sigma of the Gaussian each base is drawn from, in cells."""
    return BASE_SIGMA_IN_QUADRANT * (map_size / 2) / 2


def quadrant_cells(quadrant: int, map_size: int) -> tuple:
    """The cells of the quadrant, [low, high) on x and on y."""
    half = map_size // 2
    halves = ((0, half), (half, map_size))
    return halves[quadrant % 2], halves[quadrant // 2]


def sample_quadrant_bases(seed: int, number_of_bases: int, map_size: int) -> tuple:
    """
    Bases of the run seeded by seed, (x, y) in metres, and the quadrant of each one.
    The quadrants are a random permutation of the 4, the first number_of_bases of it.
    Each base is then drawn, one after the other from the same stream, from a Gaussian
    centred on its quadrant, snapped to the nearest cell centre and drawn again when the
    cell is outside the quadrant. The permutation takes the same draws whatever the
    number of bases, so the first bases of a seed do not change when more are added.
    """
    if not 1 <= number_of_bases <= len(QUADRANT_NAMES):
        raise ValueError(f"{number_of_bases} bases, one per quadrant: between 1 and {len(QUADRANT_NAMES)}")
    rng = np.random.default_rng(seed)
    quadrants = [int(q) for q in rng.permutation(len(QUADRANT_NAMES))[:number_of_bases]]
    sigma = base_sigma(map_size)
    map_center_offset = map_size * DISTANCE_BETWEEN_CELLS / 2
    positions = []
    for quadrant in quadrants:
        bounds = quadrant_cells(quadrant, map_size)
        ##### Centre of the quadrant in cell coordinates, between two cells when its side is even #####
        centre = [(low + high - 1) / 2 for low, high in bounds]
        while True:
            cell = np.round(rng.normal(centre, sigma))
            if all(low <= c < high for c, (low, high) in zip(cell, bounds)):
                break
        positions.append(tuple(float(c) * DISTANCE_BETWEEN_CELLS - map_center_offset for c in cell))
    return positions, quadrants


def run_job(job: dict) -> dict:
    """
    One run, in a process of its own. The scenario is set on the module constants of the
    system first: every process is fresh (maxtasksperchild=1), so they cannot reach
    another run. Then it is the run of Analytical/multiple_runs.py.
    """
    try:
        main = system_module(job["system"])
        ##### The protocol shares one centre offset between both axes, the map has to be square #####
        main.MAP_WIDTH = main.MAP_HEIGHT = job["map_size"]
        main.SIMULATION_DURATION = job["duration"]
        main.TRANSMISSION_RANGE = job["transmission_range"]
    except Exception:
        record = {"run": job["run"], "seed": job["seed"], "error": traceback.format_exc()}
    else:
        record = run_simulation(job)
    record["scenario"] = job["scenario"]
    record["base_quadrants"] = [QUADRANT_NAMES[q] for q in job["base_quadrants"]]
    return {"directory": job["directory"], "system": job["system"], "record": record}


def run_jobs(jobs: list, workers: int, configs: dict) -> int:
    """Runs the jobs in parallel and writes each run as soon as it finishes. Returns how many failed."""
    failed = 0
    ##### Forked, as in Analytical/multiple_runs.py, one process per run #####
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=workers, maxtasksperchild=1) as pool:
        for done, result in enumerate(pool.imap_unordered(run_job, jobs), start=1):
            record, directory = result["record"], result["directory"]
            name = f"{result['system'].split('/')[-1]} {record['scenario']} run {record['run']} (seed {record['seed']}"
            if "error" in record:
                failed += 1
                print(f"[{done}/{len(jobs)}] {name}) failed:\n{record['error']}", flush=True)
                continue
            path = os.path.join(directory, run_file(record["run"]))
            with open(path, "wb") as f:
                pickle.dump(record, f)
            stats = run_statistics(record, time_grid(configs[directory]))
            bases = " ".join(f"{quadrant} {format_position(position)}"
                             for quadrant, position in zip(record["base_quadrants"], record["base_positions"]))
            print(f"[{done}/{len(jobs)}] {name}, bases {bases}): mean global uncertainty "
                  f"{stats['mean_uncertainty']:.1f}, more than Nc = {record['nc_charging_base']} charging "
                  f"{stats['above_nc']:.1f}% of the time, {stats['dead']} dead, "
                  f"{record['elapsed'] / 60:.1f} min -> {path}", flush=True)
    return failed


def different_scenario(directory: str, config: dict):
    """
    The keys config.json of the directory differs on from config, when the directory holds
    runs. The runs of a directory are aggregated together, so they have to share the scenario.
    """
    config_path = os.path.join(directory, CONFIG_FILE)
    if not os.path.exists(config_path):
        return []
    if not any(RUN_PATTERN.match(name) for name in os.listdir(directory)):
        return []
    with open(config_path) as f:
        previous = json.load(f)
    ##### Compared as JSON, the way it was read back #####
    current = json.loads(json.dumps(config))
    return [key for key in current if previous.get(key) != current[key]]


def add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    """--cells, --drones and --bases, shared with run_all_tests_with_multiple_runs_bases.py."""
    parser.add_argument("--cells", type=int, default=MAP_SIZE, metavar="SIDE",
                        help=f"side of the square map in cells, {DISTANCE_BETWEEN_CELLS} m apart: "
                             "--cells 200 is a 200x200 map (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=NUMBER_OF_DRONES,
                        help="number of drones (default: %(default)s)")
    parser.add_argument("--bases", "--base", type=int, default=NUMBER_OF_BASES, dest="bases",
                        metavar="NUMBER_OF_BASES",
                        help=f"number of recharge bases, one per quadrant of the map, between 1 and "
                             f"{len(QUADRANT_NAMES)} (default: %(default)s)")


def check_scenario_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.cells < 2:
        parser.error(f"--cells {args.cells}: the map needs at least 2x2 cells, one row and column per half")
    if args.drones < 1:
        parser.error("--drones has to be at least 1")
    if not 1 <= args.bases <= len(QUADRANT_NAMES):
        parser.error(f"--bases {args.bases}: one base per quadrant, between 1 and {len(QUADRANT_NAMES)}")


def system_individual(system: str, transmission_range: float, hand_tuned: bool) -> tuple:
    """
    (individual, where it comes from) of an expanding_testing system at transmission_range:
    the one Analytical/<system> runs, see all_tests_distance.system_individuals. Raises
    ValueError when there is none or when its genes do not fit the system.
    """
    individual, origin = system_individuals(source_system(system), [transmission_range],
                                            hand_tuned)[transmission_range]
    genes = len(gene_names(system))
    if len(individual) != genes:
        raise ValueError(f"{origin}: the individual has {len(individual)} genes, {system} takes {genes}")
    return [float(gene) for gene in individual], origin


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Many runs of the expanding_testing systems on one multi-base scenario, "
                    "each run with its bases in random quadrants")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", required=True,
                        help="system(s) to run, from expanding_testing")
    add_scenario_arguments(parser)
    parser.add_argument("--range", type=float, default=TRANSMISSION_RANGE, dest="transmission_range",
                        help="communication range of the drones and of the bases, in m. Also picks the "
                             "tuned individual (default: %(default)s)")
    parser.add_argument("--hand-tuned", "--hand_tuned", action="store_true",
                        help="run HAND_TUNED_INDIVIDUAL of Analytical/<system>/main_test.py instead of the "
                             f"tuned one, into {HAND_TUNED_LOGS_ROOT}/<range>/<scenario> by default")
    parser.add_argument("--runs", type=int, default=NUMBER_OF_RUNS,
                        help="runs of every system (default: %(default)s)")
    parser.add_argument("--duration", type=int, default=SIMULATION_DURATION,
                        help="simulated seconds of each run (default: %(default)s)")
    parser.add_argument("--sample-interval", type=float, default=GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map, the "
                             "step of the series of every run (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="run k is seeded with SEED + k, which also draws its bases (default: %(default)s)")
    parser.add_argument("--logs-root",
                        help="directory, relative to each system directory, the runs are written to, "
                             f"as <range>/<scenario> (default: {DEFAULT_LOGS_ROOT}, "
                             f"{HAND_TUNED_LOGS_ROOT} with --hand-tuned)")
    parser.add_argument("--workers", type=int,
                        help="runs simulated at once (default: one per run, at most one per CPU)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the runs already in the directories and run them all again")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs has to be at least 1")
    check_scenario_arguments(parser, args)
    if args.logs_root is None:
        args.logs_root = HAND_TUNED_LOGS_ROOT if args.hand_tuned else DEFAULT_LOGS_ROOT
    systems = [SYSTEMS[name] for name in dict.fromkeys(args.system)]
    scenario = scenario_name(args.cells, args.drones, args.bases)

    ##### The individual of every system, before anything is run. No fallback to another one, #####
    ##### the range would run untuned.                                                         #####
    individuals, origins, errors = {}, {}, []
    for system in systems:
        try:
            individuals[system], origins[system] = system_individual(system, args.transmission_range,
                                                                     args.hand_tuned)
        except ValueError as error:
            errors.append(f"{system}: {error}")
    if errors:
        parser.error("no individual, nothing was run:\n" + "\n".join(errors) +
                     "\nTune the range first with Analytical/all_optimizations_distance.py, or use --hand-tuned")

    charging_capacity = system_module(systems[0]).default_charging_capacity(args.drones)
    configs = {}
    for system in systems:
        config = {
            "system": system,
            "scenario": scenario,
            "drones": args.drones,
            "transmission_range": args.transmission_range,
            "duration": args.duration,
            "sample_interval": args.sample_interval,
            "seed": args.seed,
            "individual": individuals[system],
            "map_size": args.cells,
            "bases": args.bases,
            "charging_capacity": charging_capacity,
            "base_placement": "quadrants",
            "base_sigma": base_sigma(args.cells) * DISTANCE_BETWEEN_CELLS,
            ##### Not in the runs before 2026-11-18 (1.0 or 0.5, kappa = K), so they are never mixed with these #####
            "initial_uncertainty": system_module(system).INITIAL_UNCERTAINTY,
        }
        if args.hand_tuned:
            config["hand_tuned"] = True
        configs[scenario_directory(system, args.logs_root, args.transmission_range, scenario)] = config

    ##### Every directory is checked before any is touched, so a refused one leaves the others as they were #####
    if not args.overwrite:
        refused = [f"{directory} (different {', '.join(different)})"
                   for directory, config in configs.items() if (different := different_scenario(directory, config))]
        if refused:
            parser.error("these directories hold runs of another configuration:\n  " + "\n  ".join(refused) +
                         "\nGive another --logs-root, or --overwrite to replace them")

    jobs = []
    for directory, config in configs.items():
        os.makedirs(directory, exist_ok=True)
        if args.overwrite:
            for name in os.listdir(directory):
                if RUN_PATTERN.match(name) or name in (CONFIG_FILE, SUMMARY_FILE):
                    os.remove(os.path.join(directory, name))
        with open(os.path.join(directory, CONFIG_FILE), "w") as f:
            json.dump(config, f, indent=2)
        for run in range(args.runs):
            if os.path.exists(os.path.join(directory, run_file(run))):
                continue
            seed = args.seed + run
            positions, quadrants = sample_quadrant_bases(seed, config["bases"], config["map_size"])
            jobs.append({"system": config["system"], "scenario": config["scenario"], "directory": directory,
                         "run": run, "seed": seed, "individual": config["individual"],
                         "drones": config["drones"], "map_size": config["map_size"], "bases": config["bases"],
                         "duration": args.duration, "transmission_range": args.transmission_range,
                         "sample_interval": args.sample_interval,
                         "charging_capacity": config["charging_capacity"],
                         "base_positions": positions, "base_quadrants": quadrants})
    jobs.sort(key=lambda job: job["run"])

    print(f"{', '.join(systems)}: {args.runs} runs of {scenario} ({scenario_label(scenario)}, "
          f"Nc = {charging_capacity}), {args.transmission_range:g} m, {args.duration} s, seeds from {args.seed}")
    for system in systems:
        print(f"Individual of {system} ({origins[system]}): {individuals[system]}")
    print(f"Logs: <system>/{args.logs_root}/{args.transmission_range:g}/{scenario}")
    skipped = len(configs) * args.runs - len(jobs)
    if skipped:
        print(f"{skipped} run(s) already in the directories, skipped")

    failed = 0
    if jobs:
        workers = args.workers or min(len(jobs), os.cpu_count() or 1)
        print(f"Running {len(jobs)} run(s), {workers} at once ...", flush=True)
        failed = run_jobs(jobs, workers, configs)

    lines = [f"{'system':<28} {'runs':>4}  {'mean uncertainty':>18}  {'above Nc (%)':>14}  {'dead at the end':>15}"]
    for directory in configs:
        config, runs = load_runs(directory)
        if not runs:
            continue
        with open(os.path.join(directory, SUMMARY_FILE), "w") as f:
            f.write(summary(config, runs) + "\n")
        grid = time_grid(config)
        statistics = [run_statistics(record, grid) for record in runs]

        def values(name):
            return [stats[name] for stats in statistics]

        lines.append(f"{config['system']:<28} {len(runs):>4}  "
                     f"{mean_std(values('mean_uncertainty')):>18}  {mean_std(values('above_nc')):>14}  "
                     f"{mean_std(values('dead'), 2):>15}")
    print(f"\nMean +- std deviation over the runs, {scenario}, {args.transmission_range:g} m "
          f"(every run in {SUMMARY_FILE} of its directory):")
    print("\n".join(lines))
    print(f"\nPlot them with: python3 Analytical/expanding_testing/plot_multiple_runs_bases.py "
          f"--range {args.transmission_range:g} --cells {args.cells} --drones {args.drones} --bases {args.bases}" +
          (" --hand-tuned" if args.hand_tuned else "") +
          (f" --logs-root {args.logs_root}"
           if args.logs_root != (HAND_TUNED_LOGS_ROOT if args.hand_tuned else DEFAULT_LOGS_ROOT) else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
