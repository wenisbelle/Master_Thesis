"""
Checks that the MARL environment runs the protocol of Analytical/swarm.

The environment is run twice, every decision being the one of the analytical fitness
function:
    backend   decision_backend "analytical": the environment takes the decisions itself
              (decision request, stop of the simulation, action index, apply_decision);
    policy    the decisions are the actions of a policy (env_check.AnalyticalActions)
              given through the orchestrator, the path of the neural network.
The analytical system (Analytical/swarm/main.py) is run with the same individual, seed,
base, swarm size and range. All must give the same run, bit for bit: the same global
uncertainty, dead and charging drones at every sample, the same encounters, the same
final state of every drone and the same cost.

It also checks that the scenario constants of config.py are the ones of
Analytical/swarm/main.py.

    python3 check_equivalence.py                       # a few short cases
    python3 check_equivalence.py --duration 10000 --drones 10 --range 2000 --seeds 1 2

Run it from this directory, with gradysim on the PYTHONPATH. The exit status is 0
when everything matches, so it can be used as a regression gate after changing
either protocol.
"""

import os
import sys
import time
import argparse

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import config
from env.mapping_environment import MappingEnvironment
from env.gradysim_environment.protocol import Drone
from simulation_orchestrator import AsyncMARLOrchestrator
from env_check import AnalyticalActions

##### Constants of config.py that have to be the ones of Analytical/swarm/main.py #####
SCENARIO_CONSTANTS = ("MAP_WIDTH", "MAP_HEIGHT", "UNCERTAINTY_RATE", "VANISHING_UPDATE_TIME",
                      "DRONE_SPEED", "DRONE_MAX_ACCELERATION", "DISCHARGE_RATE", "DISTANCE_BETWEEN_CELLS",
                      "KERNEL_N_SIGMA", "INITIAL_UNCERTAINTY", "MIN_CHARGE_MARGIN",
                      "GLOBAL_MAP_SAMPLE_INTERVAL", "TRAINING_DRONE_COUNTS", "BASE_POSITION_SIGMA",
                      "BEST_INDIVIDUAL", "CHARGING_BASE_POSITION", "TRANSMISSION_RANGE")
COST_CONSTANTS = {"uncertainty_weight": "UNCERTAINTY_WEIGHT", "dead_drone_penalty": "DEAD_DRONE_PENALTY",
                  "charging_congestion_penalty": "CHARGING_CONGESTION_PENALTY",
                  "charging_congestion_steepness": "CHARGING_CONGESTION_STEEPNESS"}


def check_constants(analytical) -> list:
    problems = []
    for name in SCENARIO_CONSTANTS:
        mine, theirs = getattr(config, name), getattr(analytical, name)
        if (list(mine) if isinstance(mine, (list, tuple)) else mine) != \
                (list(theirs) if isinstance(theirs, (list, tuple)) else theirs):
            problems.append(f"{name}: config.py {mine}, Analytical/swarm/main.py {theirs}")
    for field, name in COST_CONSTANTS.items():
        if getattr(config.COST, field) != getattr(analytical, name):
            problems.append(f"{name}: config.py {getattr(config.COST, field)}, "
                            f"Analytical/swarm/main.py {getattr(analytical, name)}")
    if [name for name, _, _ in analytical.GENE_BOUNDS] != list(config.GENE_NAMES):
        problems.append("GENE_NAMES differ from the GENE_BOUNDS of Analytical/swarm/main.py")
    return problems


def run_analytical(analytical, individual, drones, seed, base, duration, transmission_range):
    analytical.SIMULATION_DURATION = duration
    analytical.TRANSMISSION_RANGE = transmission_range
    analytical_protocol = sys.modules["Analytical.swarm.protocol"]
    analytical_protocol.Drone.Number_of_Encounters = 0
    results, monitor = analytical.create_and_run_simulation(
        individual, mode="train", number_of_drones=drones, seed=seed, charging_base_position=base)
    return {
        "times": np.array(monitor.times),
        "uncertainty": np.array(monitor.uncertainty),
        "dead": np.array(monitor.dead_drones),
        "charging": np.array(monitor.charging_drones),
        "unvisited": np.array(monitor.unvisited),
        "encounters": analytical_protocol.Drone.Number_of_Encounters,
        "results": results,
        "cost": float(analytical.evaluate_simulation_cost(results, monitor)),
    }


def run_marl(individual, drones, seed, base, duration, transmission_range, path="backend"):
    env = MappingEnvironment(config.scenario_config(
        num_agents_choices=(drones,),
        seed=seed,
        charging_base_position=base,
        max_episode_length=duration,
        communication_range=transmission_range,
        end_when_all_dead=False,
        decision_backend="analytical" if path == "backend" else "policy",
        analytical_parameters=config.unpack_individual(individual),
    ))
    if path == "backend":
        td = env.reset()
        while True:
            td = env.step(env.rand_action(td))
            if td["next", "done"].item():
                break
            td = td["next"]
    else:
        orchestrator = AsyncMARLOrchestrator(env, AnalyticalActions(env), collect_transitions=False)
        td = orchestrator.reset()
        while True:
            td = orchestrator.step(td)
            if td["next", "done"].item():
                break
            td = td["next"]
    monitor = env.monitor
    run = {
        "times": np.array(monitor.times),
        "uncertainty": np.array(monitor.uncertainty),
        "dead": np.array(monitor.dead_drones),
        "charging": np.array(monitor.charging_drones),
        "unvisited": np.array(monitor.unvisited),
        "encounters": Drone.Number_of_Encounters,
        "results": None,
        "cost": monitor.cost(),
        "decisions": dict(env.decisions),
    }
    env.close()
    run["results"] = env.results_aggregator
    return run


def compare(analytical_run: dict, marl_run: dict) -> list:
    problems = []
    for name in ("times", "uncertainty", "dead", "charging", "unvisited"):
        a, m = analytical_run[name], marl_run[name]
        if a.shape != m.shape:
            problems.append(f"{name}: {a.shape[0]} samples in the analytical run, {m.shape[0]} in the MARL one")
            continue
        different = np.flatnonzero(a != m)
        if different.size:
            k = different[0]
            problems.append(f"{name}: {different.size} samples differ, the first at t={analytical_run['times'][k]:g} "
                            f"(analytical {a[k]}, MARL {m[k]})")
    if analytical_run["encounters"] != marl_run["encounters"]:
        problems.append(f"encounters: analytical {analytical_run['encounters']}, MARL {marl_run['encounters']}")
    if analytical_run["results"] != marl_run["results"]:
        for drone in sorted(analytical_run["results"]):
            if analytical_run["results"][drone] != marl_run["results"].get(drone):
                problems.append(f"drone {drone} final state: analytical {analytical_run['results'][drone]}, "
                                f"MARL {marl_run['results'].get(drone)}")
    if analytical_run["cost"] != marl_run["cost"]:
        problems.append(f"cost: analytical {analytical_run['cost']!r}, MARL {marl_run['cost']!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks that the MARL environment runs the analytical protocol")
    parser.add_argument("--duration", type=int, default=2500, help="simulated seconds (default: %(default)s)")
    parser.add_argument("--drones", type=int, nargs="+", default=[3, 5], help="swarm sizes (default: %(default)s)")
    parser.add_argument("--range", type=float, nargs="+", default=[200.0, 2000.0], dest="ranges",
                        help="communication ranges, m (default: %(default)s)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="seeds (default: %(default)s)")
    parser.add_argument("--base-position", type=float, nargs=2, metavar=("X", "Y"),
                        help="recharge base, m (default: drawn from the seed as in the GA)")
    parser.add_argument("--individual", type=float, nargs=len(config.GENE_NAMES), default=config.BEST_INDIVIDUAL,
                        help="fitness parameters (default: BEST_INDIVIDUAL)")
    parser.add_argument("--paths", nargs="+", choices=("backend", "policy"), default=["backend", "policy"],
                        help="MARL paths compared with the analytical run, see the docstring (default: both)")
    args = parser.parse_args()

    from Analytical.swarm import main as analytical

    failures = 0
    problems = check_constants(analytical)
    print("Scenario constants: " + ("the same as Analytical/swarm/main.py" if not problems else "DIFFERENT"))
    for problem in problems:
        print(f"  {problem}")
    failures += bool(problems)

    from env.mapping_environment import sample_base_position
    for seed in args.seeds:
        base = tuple(args.base_position) if args.base_position else sample_base_position(
            seed, config.MAP_WIDTH, config.MAP_HEIGHT, config.DISTANCE_BETWEEN_CELLS, config.BASE_POSITION_SIGMA)
        for transmission_range in args.ranges:
            for drones in args.drones:
                start = time.time()
                analytical_run = run_analytical(analytical, list(args.individual), drones, seed, base,
                                                args.duration, transmission_range)
                elapsed = {"analytical": time.time() - start}
                for path in args.paths:
                    start = time.time()
                    marl_run = run_marl(list(args.individual), drones, seed, base, args.duration,
                                        transmission_range, path=path)
                    elapsed[path] = time.time() - start
                    problems = compare(analytical_run, marl_run)
                    failures += bool(problems)
                    print(f"[{'ok  ' if not problems else 'FAIL'}] {path:>7}: seed {seed}, {drones} drones, "
                          f"{transmission_range:g} m, base {base}, {args.duration} s: {len(marl_run['times'])} samples, "
                          f"{marl_run['encounters']} encounters, decisions {marl_run['decisions']}, "
                          f"max {int(marl_run['charging'].max())} charging, {int(marl_run['dead'][-1])} dead, "
                          f"cost {marl_run['cost']:.1f} (analytical {elapsed['analytical']:.0f} s, "
                          f"MARL {elapsed[path]:.0f} s)", flush=True)
                    for problem in problems:
                        print(f"  {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
