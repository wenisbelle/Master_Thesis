"""
Runs one system many times with the same individual, each run with its recharge
base(s) somewhere else, and keeps the swarm level series of every run, so
plot_multiple_runs.py can aggregate them.

    python3 Analytical/multiple_runs.py --system egocentric --runs 20
    python3 Analytical/multiple_runs.py --system swarm --runs 10 --drones 5 --range 500
    python3 Analytical/multiple_runs.py --system egocentric --runs 50 --drones 3 6 9 12 --range 100
    python3 Analytical/multiple_runs.py --system expanding_testing/swarm --runs 10 --bases 2
    python3 Analytical/multiple_runs.py --system expanding_testing/egocentric --runs 10 --bases 3 --map-size 150 \
        --logs-directory multiple_runs/3_bases_150

It can be started from any directory, with gradysim on the PYTHONPATH (it is in the
container). The runs are written to <system>/multiple_runs/<range>/<drones>, the same
layout as the test_logs of all_tests_distance.py:
    run_<k>.pkl    the global uncertainty, the drones charging and the dead drones at every
                   --sample-interval, plus the base(s), the seed, the cost (not for
                   expanding_testing, which has none), the final state of every drone and,
                   per cell, the times between two visits (see RevisitMonitor)
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
overrides it. centralized and naive have no logbooks per range yet, their main.py picks the
individual (default_individual): their own logbook once they are tuned, until then the
coverage genes of the swarm (centralized) or egocentric (naive) individual of the range.
Centralized has no radio, the range only files its runs and picks its individual.

--hand-tuned runs HAND_TUNED_INDIVIDUAL of <system>/main_test.py, the individual filled in
by hand, into multiple_runs_hand_tuned/<range>/<drones>, so the hand-tuned runs never mix
with the tuned ones. plot_multiple_runs.py --hand-tuned reads them back. After editing the
individual, add --overwrite: a directory refuses runs of another individual.

    python3 Analytical/multiple_runs.py --system swarm --hand-tuned --runs 10 --drones 5 --overwrite

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
from all_tests_distance import tuned_ranges, tuned_individuals, hand_tuned_individual, RANGES_OF

SYSTEMS = ("swarm", "egocentric", "centralized", "naive",
           "expanding_testing/swarm", "expanding_testing/egocentric", "expanding_testing/naive")
##### The MARL systems write the same run files with src/MARL/<system>/multiple_runs.py. They are #####
##### only read here, by plot_multiple_runs.py, so they can be plotted next to the analytical ones. #####
MARL_SYSTEMS = ("MARL/CTDE_MADQN", "MARL/Independent_MADQN")

ANALYTICAL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(ANALYTICAL_DIR)
##### The systems are imported as packages, Analytical.<system>.main, so src has to be on the path #####
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
##### The customized gradysim of the repo, ahead of any gradysim installed in a venv: an older #####
##### installed copy would run the runs silently with an older battery plugin #####
GRADYSIM_DIR = os.path.join(SRC_DIR, "gradys-sim-nextgen")
if GRADYSIM_DIR not in sys.path:
    sys.path.insert(0, GRADYSIM_DIR)

##### Defaults of main_test.py and main.py, the scenario the single tests run #####
NUMBER_OF_RUNS = 50
NUMBER_OF_DRONES = 10
SIMULATION_DURATION = 30000
TRANSMISSION_RANGE = 100
GLOBAL_MAP_SAMPLE_INTERVAL = 1.0
DISTANCE_BETWEEN_CELLS = 20

##### Spread of the bases the GA tuned on, main.BASE_POSITION_SIGMA on its 100 cell map: 3 sigma #####
##### reaches the border. It is scaled with the map, whose size expanding_testing can change.   #####
BASE_POSITION_SIGMA = 333.0
BASE_POSITION_SIGMA_MAP_SIZE = 100

DEFAULT_LOGS_DIRECTORY = "multiple_runs"
##### The runs of --hand-tuned, apart from the ones of the tuned individuals #####
HAND_TUNED_LOGS_DIRECTORY = "multiple_runs_hand_tuned"
CONFIG_FILE = "config.json"
SUMMARY_FILE = "summary.txt"
RUN_PATTERN = re.compile(r"run_(\d+)\.pkl$")
##### Series of GlobalMapMonitor kept on every sample, under the names the runs store them #####
SERIES = {"global_uncertainty": "uncertainty", "charging": "charging_drones", "dead": "dead_drones",
          ##### Drones in the waiting room of the base (since 2026-11-24, see recharge_base.py).    #####
          ##### Older runs and the MARL ones do not have it.                                       #####
          "waiting": "waiting_drones"}

##### Period of the camera timer of every protocol, in s: a drone passing over a cell sees it once #####
##### per period, so observations this close belong to the same visit.                            #####
CAMERA_PERIOD = 1.0


def system_dir(system: str) -> str:
    if system in MARL_SYSTEMS:
        return os.path.join(SRC_DIR, *system.split("/"))
    return os.path.join(ANALYTICAL_DIR, *system.split("/"))


def system_module(system: str):
    return importlib.import_module("Analytical." + system.replace("/", ".") + ".main")


def is_expanding(system: str) -> bool:
    return system.startswith("expanding_testing/")


def tuned_system(system: str) -> str:
    """
    System whose GA logbooks give the ranges and the individuals: expanding_testing has no GA
    of its own. centralized and naive pick their individual themselves (default_individual of
    their main.py), their ranges are those of swarm and egocentric (all_tests_distance.RANGES_OF).
    """
    system = system.split("/")[-1]
    return RANGES_OF.get(system, system)


def has_cost(system: str) -> bool:
    """expanding_testing has no GA and no cost, its runs are only looked at through the series."""
    return not is_expanding(system)


def gene_names(system: str) -> list:
    """Names of the genes of an individual of the system, in the order it stores them."""
    main = system_module(system)
    if is_expanding(system):
        return list(main.GENE_NAMES)
    return [name for name, _, _ in main.GENE_BOUNDS]


def default_logs_directory(transmission_range: float, number_of_drones: int, hand_tuned: bool = False) -> str:
    root = HAND_TUNED_LOGS_DIRECTORY if hand_tuned else DEFAULT_LOGS_DIRECTORY
    return os.path.join(root, f"{transmission_range:g}", str(number_of_drones))


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


class RevisitMonitor:
    """
    Per cell, the times between two visits over the run, sampled next to GlobalMapMonitor.

    Every protocol keeps in map[:, :, 1] the last time the cell was seen, 0 before it ever
    was. Only the camera of the drone over the cell writes it, the encounters only copy the
    newer time of the other drone, so the maximum over the drones is the last time any
    drone saw the cell. Every system marks a visit this way, swarm and egocentric, with
    one base or several, so the same monitor reads all of them without touching them.

    A visit lasts while the cell keeps being seen: a drone passing over it sees it on a few
    camera ticks in a row, a drone hovering over it on every one. The cell is visited again
    when it is seen after more than same_visit_gap without being seen, and the time between
    the two visits is that time without being seen, from the last time the first visit saw
    the cell to the first time the second one does: the time its uncertainty grew. It is
    read once per sample, so it can be over by up to a sample interval (when two drones see
    the cell between two samples, only the later one is read), and same_visit_gap leaves a
    sample interval of slack above the camera period.

    The intervals are not kept one by one, only their number, sum and sum of squares per
    cell, enough for the mean and the std deviation of any group of cells. The time from
    the start to the first visit, and from the last visit to the end, are not intervals
    between two visits and are left out: a cell seen at most once has none.
    """

    def __init__(self, simulation, number_of_drones: int, sample_interval: float):
        self.sim = simulation
        self.number_of_drones = number_of_drones
        self.same_visit_gap = sample_interval + CAMERA_PERIOD
        self._protocols = None
        self.last_seen = None

    def sample(self, time: float) -> None:
        if self._protocols is None:
            ##### Same nodes as GlobalMapMonitor: the drones are the first number_of_drones nodes #####
            self._protocols = [self.sim.get_node(i).protocol_encapsulator.protocol
                               for i in range(self.number_of_drones)]
        ##### Read from every protocol on every sample: an encounter replaces the map array #####
        seen = self._protocols[0].map[:, :, 1].copy()
        for protocol in self._protocols[1:]:
            np.maximum(seen, protocol.map[:, :, 1], out=seen)

        if self.last_seen is None:
            self.last_seen = np.zeros_like(seen)
            self.count = np.zeros(seen.shape, dtype=np.int32)
            self.total = np.zeros_like(seen)
            self.total_squares = np.zeros_like(seen)

        ##### Seen since the last sample. last_seen > 0: it had been seen before, so it is not its first visit #####
        new = seen > self.last_seen
        gap = seen - self.last_seen
        revisited = new & (self.last_seen > 0) & (gap > self.same_visit_gap)
        self.count[revisited] += 1
        self.total[revisited] += gap[revisited]
        self.total_squares[revisited] += gap[revisited] ** 2
        self.last_seen[new] = seen[new]

    def result(self) -> dict:
        """What the run file keeps, per cell, indexed [x, y] as the maps of the protocols."""
        return {
            "same_visit_gap": self.same_visit_gap,
            "count": self.count,
            "total": self.total,
            "total_squares": self.total_squares,
            "last_seen": self.last_seen,
        }


def cell_distance_to_base(shape: tuple, base_positions) -> np.ndarray:
    """
    Distance from the centre of every cell to the nearest base, in m, indexed [x, y]. The
    cells are placed as in the protocols: cell x is at x * DISTANCE_BETWEEN_CELLS minus half
    the side of the map, the map is centred on (0, 0). With several bases it is the base
    nearest to the cell, the one the energy term of expanding_testing uses.
    """
    width, height = shape
    x = np.arange(width) * DISTANCE_BETWEEN_CELLS - width * DISTANCE_BETWEEN_CELLS / 2
    y = np.arange(height) * DISTANCE_BETWEEN_CELLS - height * DISTANCE_BETWEEN_CELLS / 2
    return np.min([np.hypot(x[:, None] - bx, y[None, :] - by) for bx, by, *_ in base_positions], axis=0)


def mean_revisit(record: dict) -> float:
    """Mean time between two visits of a cell over every revisit of the run, NaN when there is none."""
    count = record["revisit"]["count"].sum()
    return float(record["revisit"]["total"].sum() / count) if count else float("nan")


def run_simulation(job: dict) -> dict:
    """
    One run, in a process of its own (see run_jobs). It returns what is written to
    the run file, or the traceback when the simulation crashed, so the other runs go on.
    """
    try:
        main = system_module(job["system"])
        start = time.time()
        ##### Built with the simulation, sampled with the global map #####
        monitors = []

        def revisit_observer(simulation):
            monitors.append(RevisitMonitor(simulation, job["drones"], job["sample_interval"]))
            return monitors[-1].sample

        if is_expanding(job["system"]):
            results, global_map = main.create_and_run_simulation(
                job["individual"], mode="train", sample_interval=job["sample_interval"],
                number_of_drones=job["drones"], seed=job["seed"],
                charging_base_positions=job["base_positions"],
                charging_capacity=job["charging_capacity"],
                observer_factory=revisit_observer)
        else:
            results, global_map = main.create_and_run_simulation(
                job["individual"], mode="train", sample_interval=job["sample_interval"],
                number_of_drones=job["drones"], seed=job["seed"],
                charging_base_position=job["base_positions"][0],
                observer_factory=revisit_observer)

        record = {
            "run": job["run"],
            "seed": job["seed"],
            "base_positions": job["base_positions"],
            "nc_charging_base": global_map.nc_charging_base,
            "times": np.array(global_map.times),
            **{name: np.array(getattr(global_map, attribute)) for name, attribute in SERIES.items()
               if hasattr(global_map, attribute)},
            "drones": results,
            "revisit": monitors[0].result(),
            "elapsed": time.time() - start,
        }
        if has_cost(job["system"]):
            record["cost"] = float(main.evaluate_simulation_cost(results, global_map))
            record["cost_terms"] = main.simulation_cost_terms(global_map)
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
    return {name: np.asarray(record[name])[index] for name in SERIES if name in record}


def run_statistics(record: dict, grid: np.ndarray) -> dict:
    series = aligned(record, grid)
    statistics = {
        "mean_uncertainty": float(series["global_uncertainty"].mean()),
        ##### Share of the run with more than Nc drones charging at once, in % #####
        "above_nc": 100.0 * float(np.mean(series["charging"] > record["nc_charging_base"])),
        "max_charging": int(series["charging"].max()),
        "dead": int(series["dead"][-1]),
    }
    ##### Time spent in the waiting room of the base, in drone*s #####
    if "waiting" in series:
        step = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
        statistics["waiting"] = float(series["waiting"].sum() * step)
    ##### The runs of expanding_testing have no cost, see has_cost #####
    if "cost" in record:
        statistics["cost"] = record["cost"]
    ##### Runs written before RevisitMonitor, and the MARL ones, do not have it #####
    if "revisit" in record:
        statistics["mean_revisit"] = mean_revisit(record)
    return statistics


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
    with_cost = has_cost(config["system"])
    ##### Runs written before the waiting room have no waiting series #####
    with_waiting = bool(statistics) and all("waiting" in stats for stats in statistics)
    nc = sorted({record["nc_charging_base"] for record in runs})
    lines = [
        f"{config['system']}: {len(runs)} runs, {config['drones']} drones, {config['transmission_range']:g} m, "
        f"{config['duration']} s, {config['bases']} base(s) on a {config['map_size']}x{config['map_size']} map, "
        f"Nc = {', '.join(str(n) for n in nc)}",
        f"Individual: {config['individual']}",
        "",
        f"{'run':>4} {'seed':>6} {'mean unc':>10} {'above Nc':>9} {'max charging':>13} {'dead':>5} "
        + (f"{'waiting s':>10} " if with_waiting else "")
        + (f"{'cost':>10}  " if with_cost else " ") + "base(s)",
    ]
    for record, stats in zip(runs, statistics):
        lines.append(f"{record['run']:>4} {record['seed']:>6} {stats['mean_uncertainty']:>10.1f} "
                     f"{stats['above_nc']:>8.1f}% {stats['max_charging']:>13} {stats['dead']:>5} "
                     + (f"{stats['waiting']:>10.0f} " if with_waiting else "")
                     + (f"{stats['cost']:>10.1f}  " if with_cost else " ")
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
    if with_waiting:
        lines.append("  time in the waiting room of the base: {:.0f} +- {:.0f} drone*s".format(*spread("waiting")))
    if all("mean_revisit" in stats for stats in statistics):
        lines.append("  time between two visits of a cell: {:.1f} +- {:.1f} s".format(*spread("mean_revisit")))
    if with_cost:
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
            cost = f"cost {stats['cost']:.1f}, " if has_cost(config["system"]) else ""
            print(f"[{done}/{len(jobs)}] run {record['run']} (seed {record['seed']}, base(s) "
                  + " ".join(format_position(p) for p in record["base_positions"])
                  + f"): mean global uncertainty {stats['mean_uncertainty']:.1f}, "
                  f"more than Nc = {record['nc_charging_base']} charging {stats['above_nc']:.1f}% of the time, "
                  f"{stats['dead']} dead, {cost}{record['elapsed'] / 60:.1f} min -> {path}",
                  flush=True)
    return failed


def run_scenario(parser, args, drones: int, map_size: int, individual: list, origin: str) -> int:
    """All the runs of one number of drones. Returns 1 when any run failed."""
    logs_directory = os.path.join(system_dir(args.system), args.logs_directory or
                                  default_logs_directory(args.transmission_range, drones, args.hand_tuned))
    config = {
        "system": args.system,
        "drones": drones,
        "transmission_range": args.transmission_range,
        "duration": args.duration,
        "sample_interval": args.sample_interval,
        "seed": args.seed,
        "individual": individual,
        "map_size": map_size,
        "bases": args.bases,
        "charging_capacity": args.charging_capacity,
    }
    ##### Only when set, so the directories written before the flag keep the same config #####
    if args.hand_tuned:
        config["hand_tuned"] = True

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
                     "drones": drones, "sample_interval": args.sample_interval,
                     "charging_capacity": args.charging_capacity,
                     "base_positions": sample_base_positions(seed, args.bases, map_size)})

    print(f"{args.system}: {args.runs} runs of {drones} drones, {args.transmission_range:g} m, "
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
                 f"--range {args.transmission_range:g} --drones {drones}"
                 + (" --hand-tuned" if args.hand_tuned else "")))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Many runs of one system, each with its recharge base(s) somewhere else")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", required=True,
                        help="which system(s) to run, several to run one after the other")
    parser.add_argument("--runs", type=int, default=NUMBER_OF_RUNS,
                        help="number of runs (default: %(default)s)")
    parser.add_argument("--drones", type=int, nargs="+", default=[NUMBER_OF_DRONES],
                        help="number of drones, several to run one scenario after the other, each in "
                             "its own directory (default: %(default)s)")
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
    parser.add_argument("--hand-tuned", "--hand_tuned", action="store_true",
                        help="run HAND_TUNED_INDIVIDUAL of <system>/main_test.py instead of the tuned one, "
                             f"into {HAND_TUNED_LOGS_DIRECTORY}/<range>/<drones> by default")
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

    systems = list(dict.fromkeys(args.system))
    if len(systems) > 1 and args.individual is not None:
        parser.error("--individual has the genes of one system, it cannot be used with several --system")
    if len(systems) > 1 and args.logs_directory:
        parser.error("--logs-directory is one directory, it cannot hold several --system")
    status = 0
    for system in systems:
        args.system = system
        status |= run_system(parser, args)
    return status


def run_system(parser, args) -> int:
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

    if args.individual is not None and args.hand_tuned:
        parser.error("--individual and --hand-tuned both give the individual, use one")
    if args.individual is not None:
        individual, origin = list(args.individual), "--individual"
    elif args.hand_tuned:
        individual, origin = hand_tuned_individual(args.system)
    elif hasattr(main_module, "default_individual"):
        ##### centralized and naive: their own logbook, or the coverage genes of the system they extend #####
        try:
            individual, origin = main_module.default_individual(args.transmission_range)
        except ValueError as error:
            parser.error(f"no individual for {args.system}, nothing was run:\n{error}\n"
                         f"Give the individual with --individual or --hand-tuned")
    else:
        ##### No fallback to BEST_INDIVIDUAL of main.py, the range would run untuned #####
        try:
            individual = tuned_individuals(tuned_system(args.system), [args.transmission_range])[
                args.transmission_range]
        except ValueError as error:
            parser.error(f"no tuned individual, nothing was run:\n{error}\n"
                         f"Tune the range first, or give the individual with --individual")
        origin = tuned_ranges(tuned_system(args.system))[args.transmission_range]
    genes = gene_names(args.system)
    if len(individual) != len(genes):
        parser.error(f"the individual has {len(individual)} genes, {args.system} takes "
                     f"{len(genes)}: {genes}")

    ##### One scenario per number of drones, each in its own directory #####
    if len(args.drones) > 1 and args.logs_directory:
        parser.error("--logs-directory is one directory, it cannot hold several --drones")
    status = 0
    for drones in args.drones:
        status |= run_scenario(parser, args, drones, map_size, individual, origin)
    return status


if __name__ == "__main__":
    sys.exit(main())
