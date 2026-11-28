"""
Checks of the centralized system. Run them from src, with gradysim on the path:

    python3 -m Analytical.centralized.check_centralized                        # everything
    python3 -m Analytical.centralized.check_centralized --only scheduler
    python3 -m Analytical.centralized.check_centralized --only run --duration 2500
    python3 -m Analytical.centralized.check_centralized --only sweep --seeds 4 --duration 4000

The exit status is 0 when every check passes and 1 otherwise.

scheduler  The slot placement of world.py (place_slots) on random small cases:
           - never later than a deadline, never earlier than the drone can arrive;
           - against the exact optimum of a time-indexed MILP (scipy's HiGHS) on integer
             cases: whenever a schedule within the capacity exists, the greedy finds one,
             with the same total wait (sum of latest arrival - start) as the optimum.
run        One seeded run, and the properties that make it the centralized system:
           - every drone holds the same map arrays as the world (one global map);
           - the map grows once per second, not once per drone: every cell's uncertainty
             is the rate times the time since it was last seen;
           - the drones send nothing to each other: the only messages are a landed
             drone's with the base node (landed, took off, the pad it gets);
           - the same seed gives the same run, recorded (main_test.py) or not;
           - the same seed gives the same first destinations as swarm (same scenario).
sweep      Many seeded runs, every swarm size, the base at the centre and drawn as in
           multiple_runs.py, in parallel: no drone dies, never more than max(1, Nc) drones
           charging, every drone recharges, every arrival keeps most of the reserve, no
           drone leaves the map.
"""

import os
import sys
import json
import time
import argparse
import multiprocessing

import numpy as np

from . import main
from . import main_test
from .protocol import DroneStatus
from .world import place_slots, most_on_base, on_base
from .recharge_base import BaseMessageType, base_node_id

##### Integer cases of the scheduler check #####
SCHEDULER_CASES = 150
SCHEDULER_SEED = 0
SLOT_LENGTH = 115  # length of the integer test cases, place_slots takes any (the runs book PLANNED_CHARGE_TIME + SLOT_GUARD = 365 s)

##### An arrival may be below the reserve by the wind of the flight home, but not by much #####
ARRIVAL_TOLERANCE = 0.03


def report(checks: list, title: str) -> bool:
    print(f"=== {title} ===")
    for passed, description in checks:
        print(f"[{'ok  ' if passed else 'FAIL'}] {description}")
    print()
    return all(passed for passed, _ in checks)


##################################### scheduler #####################################

def random_booked(rng, capacity: int) -> list:
    """Charges already decided, at most capacity on the base at once, as a real base would have."""
    booked = []
    for _ in range(rng.integers(0, 4)):
        start = int(rng.integers(-100, 400))
        if most_on_base(booked, start, SLOT_LENGTH) < capacity:
            booked.append(start)
    return booked


def random_case(rng):
    capacity = int(rng.integers(1, 4))
    deadlines = []
    for drone_id in range(int(rng.integers(1, 6))):
        flight_time = int(rng.integers(5, 140))
        latest_arrival = int(rng.integers(flight_time, 600))
        deadlines.append((drone_id, latest_arrival, flight_time))
    return capacity, deadlines, random_booked(rng, capacity)


def exact_slots(deadlines, booked, capacity: int):
    """
    The optimum of the slot problem on whole seconds: x[i, t] = 1 when drone i starts its
    charge at t, t from its earliest to its latest arrival, at most capacity charges on the
    base at every second, the latest starts possible (maximum sum of the starts). None when
    no schedule within the capacity exists.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix

    variables = [(i, t) for i, (_, latest, flight) in enumerate(deadlines) for t in range(flight, latest + 1)]
    if not variables:
        return {}
    first = min(t for _, t in variables)
    last = max(t for _, t in variables) + SLOT_LENGTH
    seconds = list(range(first, last))

    equal = lil_matrix((len(deadlines), len(variables)))
    capacity_rows = lil_matrix((len(seconds), len(variables)))
    for k, (i, t) in enumerate(variables):
        equal[i, k] = 1
        for s in range(t, t + SLOT_LENGTH):
            capacity_rows[s - first, k] = 1
    free = np.array([capacity - on_base(booked, s, SLOT_LENGTH) for s in seconds], dtype=float)

    result = milp(c=-np.array([t for _, t in variables], dtype=float),
                  constraints=[LinearConstraint(equal.tocsr(), 1, 1),
                               LinearConstraint(capacity_rows.tocsr(), -np.inf, free)],
                  integrality=np.ones(len(variables)), bounds=Bounds(0, 1))
    if result.status != 0:
        return None
    chosen = [variables[k] for k in np.flatnonzero(result.x > 0.5)]
    return {deadlines[i][0]: t for i, t in chosen}


def within_capacity(slots: dict, booked, capacity: int) -> bool:
    starts = list(booked) + list(slots.values())
    return all(most_on_base(starts[:k] + starts[k + 1:], start, SLOT_LENGTH) < capacity
               for k, start in enumerate(starts))


def check_scheduler() -> bool:
    rng = np.random.default_rng(SCHEDULER_SEED)
    checks = []

    ##### Bounds, on float cases with the tolerance of the simulation #####
    late = early = 0
    for _ in range(5000):
        capacity, deadlines, booked = random_case(rng)
        deadlines = [(i, latest + rng.random(), flight + rng.random()) for i, latest, flight in deadlines]
        plan = place_slots(0.0, deadlines, booked, SLOT_LENGTH, capacity, tolerance=15.0)
        for drone_id, latest, flight in deadlines:
            slot = plan[drone_id]
            late += slot.start > max(latest, flight) + 1e-9
            early += slot.start < flight - 1e-9
    checks.append((late == 0, f"no slot after the latest safe arrival ({late} of 5000 cases)"))
    checks.append((early == 0, f"no slot before the drone can arrive ({early} of 5000 cases)"))

    ##### Against the exact optimum, on whole seconds, with the tolerance 0. The greedy alone #####
    ##### (repair=False) is only reported: the simulation uses it with the repair.            #####
    infeasible = 0
    outcome = {repair: {"failed": 0, "same": 0, "worse": 0, "gaps": []} for repair in (False, True)}
    for _ in range(SCHEDULER_CASES):
        capacity, deadlines, booked = random_case(rng)
        exact = exact_slots(deadlines, booked, capacity)
        if exact is None:
            infeasible += 1
            continue
        for repair, counts in outcome.items():
            plan = place_slots(0.0, deadlines, booked, SLOT_LENGTH, capacity, tolerance=0.0, repair=repair)
            greedy = {drone_id: slot.start for drone_id, slot in plan.items()}
            if not within_capacity(greedy, booked, capacity):
                counts["failed"] += 1
                continue
            gap = sum(exact.values()) - sum(greedy.values())
            if gap <= 1e-6:
                counts["same"] += 1
            else:
                counts["worse"] += 1
                counts["gaps"].append(gap)
    feasible = SCHEDULER_CASES - infeasible

    def describe(counts):
        return (f"within the capacity in {feasible - counts['failed']}/{feasible} feasible cases, "
                f"as late as the optimum in {counts['same']}, earlier in {counts['worse']}"
                + (f" (by {np.mean(counts['gaps']):.0f} s of total wait on average)" if counts["gaps"] else ""))

    print(f"[info] {SCHEDULER_CASES} integer cases, {infeasible} with no schedule within the capacity at all")
    print(f"[info] greedy alone: {describe(outcome[False])}")
    checks.append((outcome[True]["failed"] == 0,
                   f"greedy with the repair finds a schedule within the capacity whenever one exists: "
                   f"{describe(outcome[True])}"))
    checks.append((outcome[False]["worse"] == 0,
                   "whenever the greedy alone finds a schedule within the capacity, it is the optimum"))
    return report(checks, "Scheduler (world.place_slots)")


######################################## run ########################################

def series_of(global_map) -> np.ndarray:
    return np.array([global_map.times, global_map.uncertainty, global_map.charging_drones,
                     global_map.dead_drones], dtype=float)


def first_destinations(system_main, seed: int, number_of_drones: int, base) -> np.ndarray:
    """goto_command of every drone right after initialize, in the given system."""
    captured = []

    def observer(simulation):
        def observe(_time):
            if not captured:
                captured.append(np.array([simulation.get_node(i).protocol_encapsulator.protocol.goto_command
                                          for i in range(number_of_drones)]))
        return observe

    duration = system_main.SIMULATION_DURATION
    system_main.SIMULATION_DURATION = 2
    try:
        system_main.create_and_run_simulation(system_main.BEST_INDIVIDUAL, number_of_drones=number_of_drones,
                                              seed=seed, charging_base_position=base, observer_factory=observer)
    finally:
        system_main.SIMULATION_DURATION = duration
    return captured[0]


def check_run(duration: int, number_of_drones: int, seed: int) -> bool:
    ##### main_test.py runs with a fixed Nc (CHARGING_BASE_CAPACITY), the unrecorded runs take the #####
    ##### same one so the recorded run compares with them. Restored for the sweep after it.         #####
    original_capacity = main.CHARGING_BASE_CAPACITY
    main.CHARGING_BASE_CAPACITY = main_test.CHARGING_BASE_CAPACITY
    try:
        return run_checks(duration, number_of_drones, seed)
    finally:
        main.CHARGING_BASE_CAPACITY = original_capacity


def run_checks(duration: int, number_of_drones: int, seed: int) -> bool:
    base = main.sample_base_position(seed)
    main.SIMULATION_DURATION = duration
    packets, sent = [], []
    shared = []

    ##### Every packet received and every communication command sent, counted by a subclass #####
    original_factory = main.drone_protocol_factory

    def counting_factory(**factory_kwargs):
        class CountingDrone(original_factory(**factory_kwargs)):
            def initialize(self):
                send = self.provider.send_communication_command
                self.provider.send_communication_command = lambda command: (sent.append(command), send(command))
                super().initialize()

            def handle_packet(self, message):
                packets.append(message)
                super().handle_packet(message)
        return CountingDrone

    def observer(simulation):
        protocols = [simulation.get_node(i).protocol_encapsulator.protocol for i in range(number_of_drones)]

        def observe(_time):
            world = protocols[0].world
            shared.append(all(p.map is world.map and p.is_cell_visited is world.is_cell_visited
                              for p in protocols))
        return observe

    start = time.time()
    main.drone_protocol_factory = counting_factory
    try:
        results, global_map = main.create_and_run_simulation(main.BEST_INDIVIDUAL, number_of_drones=number_of_drones,
                                                             seed=seed, charging_base_position=base,
                                                             observer_factory=observer)
    finally:
        main.drone_protocol_factory = original_factory
    elapsed = time.time() - start
    world = global_map.world
    checks = [(all(shared), "every drone holds the global map of the world, on every sample")]

    ##### Growth: a cell seen at t0 has rate * (last growth - t0), one never seen the initial value #####
    ##### plus rate * last growth. Off by at most one growth (camera and growth of the same instant). #####
    last_growth = world._last_growth
    seen = world.map[:, :, 1]
    expected = np.where(seen > 0, main.UNCERTAINTY_RATE * (last_growth - seen),
                        main.INITIAL_UNCERTAINTY + main.UNCERTAINTY_RATE * last_growth)
    error = np.abs(world.map[:, :, 0] - expected)
    checks.append((error.max() <= main.UNCERTAINTY_RATE * 1.0001,
                   f"the map grows once per second for the whole swarm (largest error "
                   f"{error.max() / main.UNCERTAINTY_RATE:.2f} growth steps, never-seen cells: "
                   f"{int(np.sum(seen == 0))})"))

    ##### Since 2026-11-24 the base is a node: the only messages are a landed drone's with it #####
    base_types = {BaseMessageType.LANDED_MESSAGE.value, BaseMessageType.TAKE_OFF_MESSAGE.value}
    only_base = (all(command.destination == base_node_id(number_of_drones)
                     and json.loads(command.message).get("message_type") in base_types for command in sent)
                 and all(json.loads(message).get("message_type") == BaseMessageType.PAD_MESSAGE.value
                         for message in packets))
    checks.append((only_base,
                   f"the drones send nothing to each other, only to the base (landed, took off), and receive "
                   f"only the pads it gives ({len(sent)} sent, {len(packets)} received)"))

    ##### Determinism, and the recorder does not change the run #####
    _, again = main.create_and_run_simulation(main.BEST_INDIVIDUAL, number_of_drones=number_of_drones,
                                              seed=seed, charging_base_position=base)
    checks.append((np.array_equal(series_of(global_map), series_of(again)),
                   "the same seed gives the same run (uncertainty, charging and dead series)"))
    recorded_results, _, recorded_map, _ = main_test.run_simulation(
        duration, number_of_drones, mode="train", individual=main.BEST_INDIVIDUAL,
        charging_base_position=base, seed=seed)
    checks.append((np.array_equal(series_of(global_map), series_of(recorded_map)) and recorded_results == results,
                   "a recorded run (main_test.py) is the same as an unrecorded one"))

    ##### Same scenario as swarm: the same streams give the same first destinations #####
    from ..swarm import main as swarm_main
    same_start = np.array_equal(first_destinations(main, seed, number_of_drones, base),
                                first_destinations(swarm_main, seed, number_of_drones, base))
    checks.append((same_start, "the same seed gives the same first destination of every drone as swarm"))

    title = (f"One run ({number_of_drones} drones, {duration} s, seed {seed}, base {base}, "
             f"{elapsed:.0f} s to simulate)")
    return report(checks, title)


####################################### sweep #######################################

def sweep_run(job: dict) -> dict:
    """One run of the sweep, in a process of its own. Only scalars come back."""
    try:
        main.SIMULATION_DURATION = job["duration"]
        arrivals, seen_charges = [], set()
        farthest = [0.0]
        most_charging = [0]

        def observer(simulation):
            protocols = [simulation.get_node(i).protocol_encapsulator.protocol for i in range(job["drones"])]

            def observe(_time):
                charging = 0
                for i, p in enumerate(protocols):
                    if p.drone_position is not None:
                        farthest[0] = max(farthest[0], float(np.max(np.abs(p.drone_position[:2]))))
                    if p.status == DroneStatus.CHARGING:
                        charging += 1
                        if (i, p.charging_start_time) not in seen_charges:
                            seen_charges.add((i, p.charging_start_time))
                            arrivals.append(p.battery_at_charging_start)
                most_charging[0] = max(most_charging[0], charging)
            return observe

        start = time.time()
        results, global_map = main.create_and_run_simulation(
            main.BEST_INDIVIDUAL, number_of_drones=job["drones"], seed=job["seed"],
            charging_base_position=job["base"], observer_factory=observer)
        recharges = [sum(1 for i, _ in seen_charges if i == drone) for drone in range(job["drones"])]
        early = [d["latest_arrival"] - d["slot_start"] for d in global_map.world.departures]
        return {**job,
                "dead": int(global_map.dead_drones[-1]),
                "most_charging": most_charging[0],
                "nc": global_map.nc_charging_base,
                "fewest_recharges": min(recharges),
                "lowest_arrival": float(min(arrivals)) if arrivals else float("nan"),
                "farthest": farthest[0],
                "mean_uncertainty": float(np.mean(global_map.uncertainty)),
                "moved_earlier": int(np.sum(np.array(early) > 1.0)),
                "departures": len(early),
                "elapsed": time.time() - start}
    except Exception:
        import traceback
        return {**job, "error": traceback.format_exc()}


def check_sweep(seeds: int, duration: int, drone_counts, workers: int) -> bool:
    jobs = []
    for drones in drone_counts:
        for seed in range(seeds):
            for base in ((0.0, 0.0), main.sample_base_position(seed)):
                jobs.append({"drones": drones, "seed": seed, "base": base, "duration": duration})

    print(f"Sweep: {len(jobs)} runs of {duration} s, {workers} at once ...", flush=True)
    with multiprocessing.get_context("fork").Pool(processes=workers, maxtasksperchild=1) as pool:
        rows = sorted(pool.map(sweep_run, jobs), key=lambda r: (r["drones"], r["seed"], r["base"]))

    print(f"{'drones':>6} {'seed':>4} {'base':>16} {'dead':>4} {'max chg':>7} {'Nc':>3} {'min rech':>8} "
          f"{'min arrival':>11} {'farthest':>8} {'earlier':>8} {'mean unc':>9} {'time':>6}")
    errors = [r for r in rows if "error" in r]
    for r in rows:
        if "error" in r:
            print(f"{r['drones']:>6} {r['seed']:>4} {str(r['base']):>16} ERROR\n{r['error']}")
            continue
        print(f"{r['drones']:>6} {r['seed']:>4} {str(r['base']):>16} {r['dead']:>4} {r['most_charging']:>7} "
              f"{r['nc']:>3} {r['fewest_recharges']:>8} {r['lowest_arrival']:>11.3f} {r['farthest']:>8.0f} "
              f"{r['moved_earlier']:>3}/{r['departures']:<4} {r['mean_uncertainty']:>9.0f} {r['elapsed']:>5.0f}s")
    print()

    good = [r for r in rows if "error" not in r]
    limit = 1.2 * main.MAP_WIDTH * main.DISTANCE_BETWEEN_CELLS / 2
    checks = [
        (not errors, f"every run finished ({len(errors)} crashed)"),
        (all(r["dead"] == 0 for r in good), f"no drone died ({sum(r['dead'] for r in good)} deaths)"),
        (all(r["most_charging"] <= max(1, r["nc"]) for r in good),
         f"never more than max(1, Nc) drones charging at once "
         f"({sum(r['most_charging'] > max(1, r['nc']) for r in good)} runs above)"),
        (all(r["fewest_recharges"] >= 1 for r in good), "every drone recharged at least once in every run"),
        (all(r["lowest_arrival"] >= main.MIN_CHARGE_MARGIN - ARRIVAL_TOLERANCE for r in good),
         f"every arrival keeps the reserve {main.MIN_CHARGE_MARGIN} up to {ARRIVAL_TOLERANCE} of wind "
         f"(lowest {min(r['lowest_arrival'] for r in good):.3f})"),
        (all(r["farthest"] <= limit for r in good), f"no drone farther than {limit:.0f} m from the map centre"),
    ]
    return report(checks, f"Sweep ({len(jobs)} runs)")


def main_checks() -> int:
    parser = argparse.ArgumentParser(description="Checks of the centralized system")
    parser.add_argument("--only", choices=("scheduler", "run", "sweep"), nargs="+",
                        default=("scheduler", "run", "sweep"), help="which checks to run (default: all)")
    parser.add_argument("--duration", type=int, default=2500,
                        help="simulated seconds of the run and of each run of the sweep (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=5, help="drones of the single run (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the single run (default: %(default)s)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds of the sweep, each with the base at the centre and drawn (default: %(default)s)")
    parser.add_argument("--sweep-drones", type=int, nargs="+", default=(3, 5, 7, 10),
                        help="swarm sizes of the sweep (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1),
                        help="runs of the sweep simulated at once (default: %(default)s)")
    args = parser.parse_args()

    passed = True
    if "scheduler" in args.only:
        passed &= check_scheduler()
    if "run" in args.only:
        passed &= check_run(args.duration, args.drones, args.seed)
    if "sweep" in args.only:
        passed &= check_sweep(args.seeds, args.duration, args.sweep_drones, args.workers)
    print("All checks passed" if passed else "SOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main_checks())
