"""
Runs the MARL system many times, each run with its recharge base somewhere else, and
keeps the swarm level series of every run: the counterpart of Analytical/multiple_runs.py,
with the same run files, so both are plotted together by Analytical/plot_multiple_runs.py.

    python3 multiple_runs.py --checkpoint checkpoints_energy/best.pt --runs 20
    python3 multiple_runs.py --checkpoint checkpoints_energy/best.pt --runs 10 --drones 5 --range 500
    python3 multiple_runs.py --policy analytical --runs 10 --logs-directory multiple_runs_analytical

and, from src:

    python3 Analytical/multiple_runs.py --system swarm --runs 10 --drones 5 --range 500 \\
        --individual 94.15 93.66 30.95 0.02 292.09 3.42 8.16 4.82 0.49
    python3 Analytical/plot_multiple_runs.py --system swarm MARL/CTDE_MADQN --range 500 --drones 5

Run it from this directory, with gradysim on the PYTHONPATH. The runs are written to
multiple_runs/<range>/<drones> (or --logs-directory), next to this file:
    run_<k>.pkl    the global uncertainty, the drones charging and the dead drones at every
                   --sample-interval, plus the base, the seed, the cost of the analytical
                   system, the final state of every drone, the decisions and the reward
    config.json    the scenario shared by every run of the directory
    summary.txt    one line per run and the mean and std deviation over them, also printed

Run k is seeded with --seed + k, which also draws its base, with the same distribution
and the same function as Analytical/multiple_runs.py. So run k of this system and run k
of the analytical one have the same base and the same wind: with the same --seed, they
are compared on the same scenarios (paired).

The runs go in parallel, a fresh process for each one. A run whose file already exists
is skipped, --overwrite runs them all again.
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
import traceback
import multiprocessing

import numpy as np

import config
from env.mapping_environment import sample_base_position

SYSTEM = "MARL/" + os.path.basename(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
POLICIES = ("nn", "analytical", "random")
DEFAULT_CHECKPOINT = os.path.join("checkpoints_energy", "best.pt")

##### Defaults of Analytical/multiple_runs.py #####
NUMBER_OF_RUNS = 10
NUMBER_OF_DRONES = 10

DEFAULT_LOGS_DIRECTORY = "multiple_runs"
CONFIG_FILE = "config.json"
SUMMARY_FILE = "summary.txt"
RUN_PATTERN = re.compile(r"run_(\d+)\.pkl$")
##### Series of the monitor kept on every sample, under the names the runs store them #####
SERIES = {"global_uncertainty": "uncertainty", "charging": "charging_drones", "dead": "dead_drones"}


def default_logs_directory(transmission_range: float, number_of_drones: int) -> str:
    return os.path.join(DEFAULT_LOGS_DIRECTORY, f"{transmission_range:g}", str(number_of_drones))


def run_file(run: int) -> str:
    return f"run_{run:03d}.pkl"


def run_simulation(job: dict) -> dict:
    """
    One run, in a process of its own (see run_jobs). It returns what is written to
    the run file, or the traceback when the simulation crashed, so the other runs go on.
    """
    try:
        import torch
        torch.set_num_threads(1)
        from main_test import run_simulation as run_test

        start = time.time()
        results, _, monitor, env = run_test(
            duration=job["duration"], number_of_drones=job["drones"], policy=job["policy"],
            checkpoint=job["checkpoint"], mode="train", sample_interval=job["sample_interval"],
            individual=job["individual"], charging_base_position=job["base_positions"][0],
            transmission_range=job["transmission_range"], seed=job["seed"], record=False)

        return {
            "run": job["run"],
            "seed": job["seed"],
            "base_positions": job["base_positions"],
            "nc_charging_base": monitor.nc_charging_base,
            "times": np.array(monitor.times),
            **{name: np.array(getattr(monitor, attribute)) for name, attribute in SERIES.items()},
            "drones": results,
            "cost": monitor.cost(),
            "cost_terms": monitor.cost_terms(),
            "decisions": dict(env.decisions),
            "reward": env.episode_metrics(),
            "elapsed": time.time() - start,
        }
    except Exception:
        return {"run": job["run"], "seed": job["seed"], "error": traceback.format_exc()}


##### Statistics of a run, the same as Analytical/multiple_runs.py #####
def time_grid(config_: dict) -> np.ndarray:
    return np.arange(0.0, config_["duration"], config_["sample_interval"])


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
        config_ = json.load(f)
    runs = []
    for name in os.listdir(directory):
        if RUN_PATTERN.match(name):
            with open(os.path.join(directory, name), "rb") as f:
                runs.append(pickle.load(f))
    return config_, sorted(runs, key=lambda record: record["run"])


def format_position(position) -> str:
    return "(" + ", ".join(f"{c:g}" for c in position) + ")"


def summary(config_: dict, runs: list) -> str:
    grid = time_grid(config_)
    statistics = [run_statistics(record, grid) for record in runs]
    nc = sorted({record["nc_charging_base"] for record in runs})
    lines = [
        f"{config_['system']}: {len(runs)} runs, {config_['drones']} drones, {config_['transmission_range']:g} m, "
        f"{config_['duration']} s, {config_['bases']} base(s) on a {config_['map_size']}x{config_['map_size']} map, "
        f"Nc = {', '.join(str(n) for n in nc)}",
        f"Policy: {config_['policy_description']}",
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


def run_jobs(jobs: list, workers: int, logs_directory: str, config_: dict) -> int:
    """Runs the jobs in parallel and writes each run as soon as it finishes. Returns how many failed."""
    failed = 0
    grid = time_grid(config_)
    ##### Spawned, not forked: torch is already loaded here. One process per run. #####
    context = multiprocessing.get_context("spawn")
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
            print(f"[{done}/{len(jobs)}] run {record['run']} (seed {record['seed']}, base "
                  + " ".join(format_position(p) for p in record["base_positions"])
                  + f"): mean global uncertainty {stats['mean_uncertainty']:.1f}, "
                  f"more than Nc = {record['nc_charging_base']} charging {stats['above_nc']:.1f}% of the time, "
                  f"{stats['dead']} dead, cost {stats['cost']:.1f}, {record['elapsed'] / 60:.1f} min -> {path}",
                  flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Many runs of the MARL system, each with its recharge base somewhere else")
    parser.add_argument("--policy", choices=POLICIES, default="nn",
                        help="who decides, see main_test.py (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="--policy nn: checkpoint of the network, relative to this directory "
                             "(default: %(default)s)")
    parser.add_argument("--individual", type=float, nargs=len(config.GENE_NAMES), default=config.BEST_INDIVIDUAL,
                        help="--policy analytical: genes of the individual (default: BEST_INDIVIDUAL)")
    parser.add_argument("--runs", type=int, default=NUMBER_OF_RUNS,
                        help="number of runs (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=NUMBER_OF_DRONES,
                        help="number of drones (default: %(default)s)")
    parser.add_argument("--range", type=float, default=config.TRANSMISSION_RANGE, dest="transmission_range",
                        help="communication range of the drones, in m (default: %(default)s)")
    parser.add_argument("--duration", type=int, default=config.SIMULATION_DURATION,
                        help="simulated seconds of each run (default: %(default)s)")
    parser.add_argument("--sample-interval", type=float, default=config.GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map, the "
                             "integration step of the cost (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="run k is seeded with SEED + k, which also draws its base (default: %(default)s)")
    parser.add_argument("--logs-directory", "--logs_directory",
                        help="directory, relative to this directory, the runs are written to "
                             "(default: multiple_runs/<range>/<drones>)")
    parser.add_argument("--workers", type=int,
                        help="runs simulated at once (default: one per run, at most one per CPU)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the runs already in the directory and run them all again")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs has to be at least 1")
    checkpoint = os.path.join(ROOT_DIR, args.checkpoint)
    if args.policy == "nn" and not os.path.exists(checkpoint):
        parser.error(f"no checkpoint {checkpoint}: train the network first (train.py), or give another --checkpoint")

    from main_test import describe_policy
    logs_directory = os.path.join(ROOT_DIR, args.logs_directory or
                                  default_logs_directory(args.transmission_range, args.drones))
    run_config = {
        "system": SYSTEM,
        "drones": args.drones,
        "transmission_range": args.transmission_range,
        "duration": args.duration,
        "sample_interval": args.sample_interval,
        "seed": args.seed,
        "policy": args.policy,
        "checkpoint": args.checkpoint if args.policy == "nn" else None,
        "individual": list(args.individual) if args.policy == "analytical" else None,
        "policy_description": describe_policy(args.policy, args.checkpoint, args.individual),
        "map_size": config.MAP_WIDTH,
        "bases": 1,
        "charging_capacity": None,
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
        current = json.loads(json.dumps(run_config))
        different = [key for key in current if previous.get(key) != current[key]]
        if different and existing:
            parser.error(f"{logs_directory} holds {len(existing)} runs of another scenario "
                         f"(different {', '.join(different)}). Give another --logs-directory, "
                         f"or --overwrite to replace them")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    jobs = []
    for run in range(args.runs):
        if os.path.exists(os.path.join(logs_directory, run_file(run))):
            continue
        seed = args.seed + run
        ##### The base of Analytical/multiple_runs.py for the same seed #####
        base = sample_base_position(seed, config.MAP_WIDTH, config.MAP_HEIGHT, config.DISTANCE_BETWEEN_CELLS,
                                    config.BASE_POSITION_SIGMA)
        jobs.append({"run": run, "seed": seed, "policy": args.policy, "checkpoint": checkpoint,
                     "individual": list(args.individual), "drones": args.drones,
                     "transmission_range": args.transmission_range, "duration": args.duration,
                     "sample_interval": args.sample_interval, "base_positions": [base]})

    print(f"{SYSTEM}: {args.runs} runs of {args.drones} drones, {args.transmission_range:g} m, "
          f"{args.duration} s, the base drawn again on every run")
    print(f"Policy: {run_config['policy_description']}")
    print(f"Logs: {logs_directory}")
    skipped = args.runs - len(jobs)
    if skipped:
        print(f"{skipped} run(s) already in the directory, skipped")

    failed = 0
    if jobs:
        workers = args.workers or min(len(jobs), os.cpu_count() or 1)
        ##### One thread per run, the runs already fill the CPUs #####
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        print(f"Running {len(jobs)} run(s), {workers} at once ...", flush=True)
        failed = run_jobs(jobs, workers, logs_directory, run_config)

    run_config, runs = load_runs(logs_directory)
    if runs:
        text = summary(run_config, runs)
        with open(os.path.join(logs_directory, SUMMARY_FILE), "w") as f:
            f.write(text + "\n")
        print()
        print(text)
        print(f"\nPlot them, from src, with: python3 Analytical/plot_multiple_runs.py --system {SYSTEM} "
              + (f"--logs-directory {args.logs_directory}" if args.logs_directory else
                 f"--range {args.transmission_range:g} --drones {args.drones}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
