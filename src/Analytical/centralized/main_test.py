"""
Run it from the src directory, with gradysim on the PYTHONPATH:

    python3 -m Analytical.centralized.main_test
    python3 -m Analytical.centralized.main_test --duration 1000 --drones 3
    python3 -m Analytical.centralized.main_test --sample-interval 0.5   # finer global map sampling
    python3 -m Analytical.centralized.main_test --map-plot        # live per-drone map (very slow)
    python3 -m Analytical.centralized.main_test --simulation-plot # gradysim 3D view
    python3 -m Analytical.centralized.main_test --logs_directory logs_3   # keep this run apart
    python3 -m Analytical.centralized.main_test --individual 94.2 93.7 31.0 292.1  # another individual
    python3 -m Analytical.centralized.main_test --encounter-radius 500   # pair decisions farther away
    python3 -m Analytical.centralized.main_test --base-position 400 -200  # recharge base off the map centre

There is no --transmission-range: centralized has no radio.

The protocol log and the recording go to the logs directory, next to this file
(logs by default). Plot it from the plots directory with:

    python3 plot_drones.py --directory logs_3

The exit status is 0 when every check passes and 1 otherwise, so it can also
be used as a regression gate.
"""

import os
import sys
import logging
import argparse

import numpy as np

from . import main
from .main import BEST_INDIVIDUAL, GENE_BOUNDS
from .recorder import SimulationRecorder
from .protocol import DroneStatus, Drone, drone_protocol_factory
from .estimation_metrics import summarize, format_summary


SIMULATION_DURATION = 30000
NUMBER_OF_DRONES = 9

##### The individual filled in by hand, one gene per line in the GENE_BOUNDS order of main.py.  #####
##### --hand-tuned runs it instead of the GA logbooks, here and in all_tests_distance.py,        #####
##### multiple_runs.py and run_all_tests_with_multiple_runs.py (plot_multiple_runs.py reads the  #####
##### runs back with --hand-tuned too). It starts as BEST_INDIVIDUAL of main.py.                 #####
HAND_TUNED_INDIVIDUAL = [
    1.0,    # base_variance, between 0.25 and 100. Same kernel in every system so they compare fairly (2026-11-24)
    1.0,    # alpha_variance_modifier, between 0 and 100. Same kernel in every system so they compare fairly (2026-11-24)
    100.0,  # energy_gamma, between 0.1 and 100
    200.0,  # distance_between_drone_norm, between 10 and 1000
]

##### How often the drone state is sampled.
MONITOR_INTERVAL = 10.0

##### Cell spacing of the protocol, needed to turn metres into cells #####
DISTANCE_BETWEEN_CELLS = 20

##### Each run is written into a logs directory next to this file, so runs can #####
##### be kept apart. The file names are fixed, plots/*.py look them up there.   #####
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGS_DIRECTORY = "logs"
LOG_FILE = "main_test.log"
RECORDING_FILE = "test_recording.pkl"

##### What made a drone choose the base, abort or die, from the timer the protocol was  #####
##### handling when its status changed. See SimulationRecorder.record_return_event. In  #####
##### centralized only the scheduler sends a drone to the base, on its mobility tick,   #####
##### and no message is ever received.                                                   #####
TIMER_TRIGGERS = {"mobility": "scheduler", "recharge_done": "after_recharge", "battery_check": "battery",
                  "encounter_check": "encounter"}


def make_recorded_drone(recorder_holder: dict, factory, **factory_kwargs):
    """
    Builds the configured protocol with the usual factory and wraps it so every
    change of destination is written to the recorder.

    The rest of the state is pulled from the simulator by the recorder itself.
    Destinations cannot be: they are decided between two samples, by three
    different paths (the drone's own choice, a pair decision, a destination set
    by the other drone of a pair decision), so they are caught here instead, by
    watching goto_command after each event the protocol handles, and when the
    other drone sets it (receive_destination). The same goes for the changes of
    status that make a trip to the base, see _record_return.
    """
    ConfiguredDrone = factory(**factory_kwargs)

    class RecordedDrone(ConfiguredDrone):
        def initialize(self) -> None:
            super().initialize()
            self._last_goto_command = np.array(self.goto_command, dtype=float)
            self._last_status = self.status

        def _record_return(self, trigger_of) -> None:
            """
            Records the drone choosing the base, reaching it, being sent back to
            mapping on the way, or dying, with the battery it has at that moment.
            trigger_of returns (trigger, sender) and is only called on one of these
            changes: it parses the packet, which costs as much as the protocol's
            own parse for a gossip carrying the map.
            """
            previous, current = self._last_status, self.status
            if previous == current:
                return
            self._last_status = current

            if current == DroneStatus.GOING_TO_BASE:
                event = "decision"
            elif current == DroneStatus.DEAD:
                event = "death"
            elif previous == DroneStatus.GOING_TO_BASE and current in (DroneStatus.CHARGING, DroneStatus.WAITING):
                ##### Landed on the base, on a free pad or in the waiting room #####
                event = "arrival"
            elif previous == DroneStatus.GOING_TO_BASE and current == DroneStatus.MAPPING:
                event = "abort"
            else:
                ##### WAITING -> CHARGING (a pad freed) and CHARGING -> MAPPING (the end #####
                ##### of the recharge) are not part of a trip                            #####
                return

            recorder = recorder_holder.get("recorder")
            if recorder is None:
                return
            trigger, sender = trigger_of()
            recorder.record_return_event(self.provider.get_id(), event, self.provider.current_time(),
                                         trigger, sender=sender, previous_status=previous.name)

        def _record_decision(self) -> None:
            goto = np.array(self.goto_command, dtype=float)
            if np.allclose(goto, self._last_goto_command):
                return
            self._last_goto_command = goto

            ##### A dead drone is parked on its own position, that is not a decision #####
            recorder = recorder_holder.get("recorder")
            if recorder is None or self.drone_position is None or self.status == DroneStatus.DEAD:
                return

            recorder.record_decision(self.provider.get_id(),
                                     self.provider.current_time(),
                                     self.drone_position,
                                     goto)

        def handle_timer(self, timer: str) -> None:
            super().handle_timer(timer)
            self._record_decision()
            self._record_return(lambda: (TIMER_TRIGGERS.get(timer, timer), None))

        def receive_destination(self, target_coords) -> None:
            ##### Set by the other drone of a pair decision, during its event, not this one's #####
            super().receive_destination(target_coords)
            self._record_decision()

    return RecordedDrone


def run_simulation(duration: int, number_of_drones: int,
                   mode: str = "test",
                   monitor_interval: float = MONITOR_INTERVAL,
                   sample_interval: float = main.GLOBAL_MAP_SAMPLE_INTERVAL,
                   enable_map_plot: bool = False,
                   enable_simulation_plot: bool = False,
                   individual=BEST_INDIVIDUAL,
                   charging_base_position: tuple = main.CHARGING_BASE_POSITION,
                   seed: int = None):
    """
    Runs a single simulation of main.py with the given individual and recharge
    base, recording the state of every drone every monitor_interval seconds.
    seed fixes the wind and the initial destinations, see
    main.create_and_run_simulation. None is a random run.

    main.py already steps the simulation to sample the global map every
    sample_interval seconds, so the recording rides on the same samples and only
    keeps one out of every monitor_interval / sample_interval of them, which is
    all the plots need.

    """
    recorder_holder = {}
    ##### Time of the last sample offered by main.py and of the last one kept, 
    ##### so the end of the run can be recorded without duplicating a sample.  
    last_observed = [None]
    last_recorded = [None]
    original_factory = main.drone_protocol_factory
    original_duration = main.SIMULATION_DURATION
    original_number_of_drones = main.NUMBER_OF_DRONES

    def recorded_factory(**factory_kwargs):
        return make_recorded_drone(recorder_holder, original_factory, **factory_kwargs)

    def make_recorder(simulation):
        recorder = SimulationRecorder(
            simulation,
            number_of_drones,
            main.MAP_WIDTH,
            main.MAP_HEIGHT,
            distance_between_cells=DISTANCE_BETWEEN_CELLS,
            status_labels={status.value: status.name for status in DroneStatus},
        )
        recorder_holder["recorder"] = recorder

        ##### Decimation of the samples main.py takes, so the recording stays 
        ##### at monitor_interval. A full map per drone is stored on each one. 
        next_sample = [0.0]

        def observe(time):
            last_observed[0] = time
            if time < next_sample[0]:
                return
            recorder.snapshot(time)
            last_recorded[0] = time
            ##### Anchored on the grid, not on the time of the sample taken: the 
            ##### first event does not land on 0 and the offset would be carried 
            ##### by every sample after it.                                      
            while next_sample[0] <= time:
                next_sample[0] += monitor_interval

        return observe

    main.drone_protocol_factory = recorded_factory
    main.SIMULATION_DURATION = duration
    main.NUMBER_OF_DRONES = number_of_drones

    try:
        results, global_map = main.create_and_run_simulation(
            individual,
            mode=mode,
            enable_map_plot=enable_map_plot,
            enable_simulation_plot=enable_simulation_plot,
            sample_interval=sample_interval,
            observer_factory=make_recorder,
            charging_base_position=charging_base_position,
            seed=seed,
        )
        cost = main.evaluate_simulation_cost(results, global_map, mode=mode)
    finally:
        main.drone_protocol_factory = original_factory
        main.SIMULATION_DURATION = original_duration
        main.NUMBER_OF_DRONES = original_number_of_drones

    ##### The run ends on the event that closes the simulation, which can fall 
    ##### before the next multiple of monitor_interval, so the final state is  
    ##### added here rather than being left out of the recording.              
    recorder = recorder_holder["recorder"]
    if last_observed[0] is not None and last_observed[0] != last_recorded[0]:
        recorder.snapshot(last_observed[0])

    return results, recorder, global_map, cost


def status_episodes(times, statuses) -> list:
    """
    Collapses the samples into the list of periods the drone spent in each
    status, as (status, start_time, end_time).
    """
    episodes = []
    for time, status in zip(times, statuses):
        status = DroneStatus(int(status))
        if episodes and episodes[-1][0] == status:
            episodes[-1][2] = time
        else:
            episodes.append([status, time, time])
    return [(status, start, end) for status, start, end in episodes]


def drone_episodes(recorder, drone_id: int) -> list:
    state = recorder.series[drone_id]
    return status_episodes(recorder.times, state["status"])


def format_episodes(episodes: list, max_episodes: int = 12) -> str:
    shown = episodes[:max_episodes]
    text = " | ".join(f"{status.name} {start:.0f}-{end:.0f}" for status, start, end in shown)
    if len(episodes) > max_episodes:
        text += f" | ... (+{len(episodes) - max_episodes} more)"
    return text


def report(results: dict, recorder: SimulationRecorder,
           global_map: main.GlobalMapMonitor, cost: float,
           duration: int, number_of_drones: int,
           monitor_interval: float = MONITOR_INTERVAL,
           individual=BEST_INDIVIDUAL,
           charging_base_position: tuple = main.CHARGING_BASE_POSITION) -> bool:
    """
    Prints the summary of the run and returns whether every check passed.
    """
    total_cells = main.MAP_WIDTH * main.MAP_HEIGHT
    map_half_extent = main.MAP_WIDTH * DISTANCE_BETWEEN_CELLS / 2

    print("=== Scenario ===")
    print(f"Duration: {duration} s, drones: {number_of_drones}, "
          f"map: {main.MAP_WIDTH}x{main.MAP_HEIGHT} cells ({total_cells} cells)")
    print(f"Uncertainty rate: {main.UNCERTAINTY_RATE} every {main.VANISHING_UPDATE_TIME} s, "
          f"no radio (global map, true state of every drone), pair decisions within "
          f"{main.ENCOUNTER_RADIUS:g} m, speed: {main.DRONE_SPEED} m/s")
    print(f"Recharge base at {tuple(charging_base_position)}, map spans "
          f"+-{map_half_extent:.0f} m on each axis")
    print("Individual: " + ", ".join(
        f"{name}={value:g}" for (name, _, _), value in zip(GENE_BOUNDS, individual)))
    print()

    print("=== Per drone ===")
    header = (f"{'id':>3} {'coverage':>9} {'final unc':>10} {'distance':>10} "
              f"{'battery':>8} {'status':>14} {'recharges':>10} {'charging s':>11} "
              f"{'decisions':>10}")
    print(header)

    recharges_per_drone = {}
    off_map_per_drone = {}

    for drone_id in sorted(recorder.series):
        episodes = drone_episodes(recorder, drone_id)
        drone_results = results[drone_id]

        recharge_episodes = [e for e in episodes if e[0] == DroneStatus.CHARGING]
        charging_time = sum(end - start for _, start, end in recharge_episodes)
        recharges_per_drone[drone_id] = len(recharge_episodes)

        ##### The position is only known once the first telemetry arrives, the #####
        ##### samples taken before it are NaN.                                  #####
        positions = np.array(recorder.world_pos[drone_id], dtype=float)
        positions = positions[~np.isnan(positions[:, 0])]
        off_map_per_drone[drone_id] = float(np.max(np.abs(positions[:, :2]))) if len(positions) else 0.0

        ##### The map is shared, so the coverage of a drone is what its own camera saw #####
        coverage = 100.0 * drone_results["own_cells_seen"] / total_cells

        print(f"{drone_id:>3} {coverage:>8.1f}% {drone_results['final_uncertainty']:>10.1f} "
              f"{drone_results['total_distance_traveled']:>9.0f}m "
              f"{drone_results['final_battery_status']:>7.2f} "
              f"{DroneStatus(drone_results['drone_status']).name:>14} "
              f"{len(recharge_episodes):>10} {charging_time:>11.0f} "
              f"{len(recorder.decisions[drone_id]):>10}")
    print()

    print(f"=== Status timeline (sampled every {monitor_interval:.0f} s) ===")
    for drone_id in sorted(recorder.series):
        print(f"drone {drone_id}: {format_episodes(drone_episodes(recorder, drone_id))}")
    print()

    ##### The global map is the element-wise minimum of the maps of every drone: #####
    ##### what the system knows as a whole, which is what the GA minimizes.      #####
    print("=== Swarm (global map) ===")
    swarm_visited = total_cells - global_map.unvisited_cells
    print(f"Cells seen by at least one drone: {swarm_visited}/{total_cells} "
          f"({100.0 * swarm_visited / total_cells:.1f}%)")
    print(f"Pair decisions (encounters within {main.ENCOUNTER_RADIUS:g} m): {Drone.Number_of_Encounters}")
    print(f"Global final uncertainty: {global_map.final_uncertainty:.1f} "
          f"(sum over the {total_cells} cells)")
    print(f"Global accumulated uncertainty: {global_map.accomulated_uncertainty:.1f} "
          f"(integrated every {global_map.sample_interval:g} s, {len(global_map.times)} samples)")
    print(f"Mean per-drone accumulated uncertainty: "
          f"{np.mean([r['accomulated_uncertainty'] for r in results.values()]):.1f}")
    print(f"Cost (the value the GA minimizes): {cost:.4f}")
    print()

    ##### What the scheduler did. A departure is a drone sent to the base, "early" is how #####
    ##### long before its latest safe arrival the slot it got starts.                     #####
    print("=== Charging (the scheduler) ===")
    nc = global_map.nc_charging_base
    departures = global_map.world.departures
    charging = np.array(global_map.charging_drones)
    arrivals = [event["battery"] for events in recorder.returns.values()
                for event in events if event["event"] == "arrival"]
    print(f"Nc: {nc}, slot guard: {global_map.world.SLOT_GUARD:g} s, reserve: {main.MIN_CHARGE_MARGIN}")
    if departures:
        early = np.array([d["latest_arrival"] - d["slot_start"] for d in departures])
        battery = np.array([d["battery"] for d in departures])
        print(f"Departures: {len(departures)}, battery at departure: mean {battery.mean():.3f}, "
              f"min {battery.min():.3f}, max {battery.max():.3f}")
        print(f"Slot before the latest safe arrival by: mean {early.mean():.0f} s, max {early.max():.0f} s, "
              f"{np.sum(early > 1.0)} of {len(early)} departures moved earlier for the capacity")
    if arrivals:
        print(f"Battery at arrival on the base: mean {np.mean(arrivals):.3f}, min {np.min(arrivals):.3f}")
    print(f"Most drones charging at once: {charging.max() if charging.size else 0}, "
          f"time with more than Nc charging: {100.0 * np.mean(charging > nc) if charging.size else 0.0:.2f}% "
          f"({global_map.accomulated_charging_excess:.0f} drone*s)")
    print()

    ##### How wrong the drones are about each other's battery, and at which battery #####
    ##### they go back. plots/plot_estimation.py and plot_returns.py draw the same. #####
    ##### In centralized the belief is the truth, the error is 0 by construction.    #####
    print("=== Battery estimation and returns to the base ===")
    status_values = {status.name: status.value for status in DroneStatus}
    print(format_summary(summarize(recorder.drone_arrays(), recorder.returns, status_values)))
    print()

    ##### The base: its pads and its waiting room, see recharge_base.py #####
    recharge_base = global_map.recharge_base
    print("=== Base (pads and waiting room) ===")
    print(recharge_base.report(global_map.times[-1] if global_map.times else duration, global_map.charging_drones))
    print()

    ##### Checks. Each one is a claim that has to hold for the run to be #####
    ##### considered healthy, printed so a failure is obvious.           #####
    print("=== Checks ===")
    checks = []

    ##### The pads are a hard limit: a drone that finds them taken waits landed #####
    most_charging = int(max(global_map.charging_drones, default=0))
    checks.append((
        most_charging <= recharge_base.pads,
        f"never more than {recharge_base.pads} drone(s) charging at once, the pads of the base "
        f"(at most {most_charging} were)"))

    mapping_drones = [drone_id for drone_id, drone_results in results.items()
                      if drone_results["own_cells_seen"] > 0]
    checks.append((
        len(mapping_drones) == len(results),
        f"every drone mapped something ({len(mapping_drones)}/{len(results)} did)"))

    checks.append((
        swarm_visited > 0,
        f"the swarm covered {100.0 * swarm_visited / total_cells:.1f}% of the map"))

    ##### The global map is a minimum, so it cannot be worse than the best drone. #####
    ##### If it is, the maps are being sampled out of sync with the protocols.    #####
    best_drone_uncertainty = min(r["final_uncertainty"] for r in results.values())
    checks.append((
        global_map.final_uncertainty <= best_drone_uncertainty + 1e-6,
        f"the global map ({global_map.final_uncertainty:.1f}) is at least as good as "
        f"the best single drone ({best_drone_uncertainty:.1f})"))

    drones_that_recharged = [d for d, n in recharges_per_drone.items() if n > 0]
    checks.append((
        len(drones_that_recharged) == len(recharges_per_drone),
        f"every drone reached the base and recharged at least once "
        f"({len(drones_that_recharged)}/{len(recharges_per_drone)} did, "
        f"{sum(recharges_per_drone.values())} recharges in total)"))

    dead_drones = [drone_id for drone_id, drone_results in results.items()
                   if drone_results["drone_status"] == DroneStatus.DEAD.value]
    checks.append((
        not dead_drones,
        f"no drone ran out of battery{'' if not dead_drones else f' (dead: {dead_drones})'}"))

    ##### The drones fly on velocity commands, so a destination that is never #####
    ##### re-aimed shows up as a drone drifting away from the map forever.    #####
    tolerance = 1.2 * map_half_extent
    far_drones = {drone_id: distance for drone_id, distance in off_map_per_drone.items()
                  if distance > tolerance}
    checks.append((
        not far_drones,
        f"every drone stayed within {tolerance:.0f} m of the map centre"
        f"{'' if not far_drones else f' (outside: {far_drones})'}"))

    ##### The old check "never more than max(1, Nc) drones charging at once" is the pad #####
    ##### check at the top since 2026-11-24: a drone the scheduler sends to a full base  #####
    ##### (no free slot left before its deadline) now waits landed, see recharge_base.py #####

    for passed, description in checks:
        print(f"[{'ok  ' if passed else 'FAIL'}] {description}")

    return all(passed for passed, _ in checks)


def main_test() -> int:
    parser = argparse.ArgumentParser(
        description="Single simulation of the analytical protocol, to check that the "
                    "drones map the area and go back to the recharge base")
    parser.add_argument("--duration", type=int, default=SIMULATION_DURATION,
                        help="simulated seconds (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=NUMBER_OF_DRONES,
                        help="number of drones (default: %(default)s)")
    parser.add_argument("--logs_directory", "--logs-directory", default=DEFAULT_LOGS_DIRECTORY,
                        help=f"directory, relative to {ROOT_DIR}, where the per-routine "
                             f"protocol log ({LOG_FILE}) and the recording read by the plot "
                             f"scripts ({RECORDING_FILE}) are written (default: %(default)s)")
    parser.add_argument("--monitor-interval", type=float, default=MONITOR_INTERVAL,
                        help="simulated seconds between two samples of the drone state "
                             "(default: %(default)s)")
    parser.add_argument("--sample-interval", type=float,
                        default=main.GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map, "
                             "the integration step of the cost (default: %(default)s)")
    parser.add_argument("--individual", type=float, nargs=len(GENE_BOUNDS),
                        metavar=tuple(name.upper() for name, _, _ in GENE_BOUNDS),
                        help="genes of the individual to run, in the GENE_BOUNDS order "
                             f"(default: BEST_INDIVIDUAL of main.py, {BEST_INDIVIDUAL})")
    parser.add_argument("--hand-tuned", "--hand_tuned", action="store_true",
                        help="run HAND_TUNED_INDIVIDUAL of this file, filled in by hand")
    parser.add_argument("--encounter-radius", type=float, default=main.ENCOUNTER_RADIUS,
                        help="distance under which two MAPPING drones make a pair decision, in m "
                             "(default: %(default)s)")
    parser.add_argument("--base-position", type=float, nargs=2, default=main.CHARGING_BASE_POSITION,
                        metavar=("X", "Y"),
                        help="position of the recharge base, in m, the map centre is (0, 0). "
                             "The drones take off from it (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="fixes the wind and the initial destinations of the drones, so "
                             "two runs can be compared (default: a random run)")
    parser.add_argument("--quiet", action="store_true",
                        help="run the protocol in train mode, without the per-routine log")
    parser.add_argument("--map-plot", action="store_true",
                        help="draw the live per-drone map (slow)")
    parser.add_argument("--simulation-plot", action="store_true",
                        help="open the gradysim visualization (slow)")
    args = parser.parse_args()
    if args.hand_tuned and args.individual is not None:
        parser.error("--hand-tuned and --individual both give the individual, use one")
    if len(HAND_TUNED_INDIVIDUAL) != len(GENE_BOUNDS):
        parser.error(f"HAND_TUNED_INDIVIDUAL has {len(HAND_TUNED_INDIVIDUAL)} genes, GENE_BOUNDS "
                     f"{len(GENE_BOUNDS)}: {[name for name, _, _ in GENE_BOUNDS]}")
    individual = (list(HAND_TUNED_INDIVIDUAL) if args.hand_tuned
                  else args.individual if args.individual is not None else list(BEST_INDIVIDUAL))

    charging_base_position = tuple(args.base_position)
    map_half_extents = (main.MAP_WIDTH * DISTANCE_BETWEEN_CELLS / 2,
                        main.MAP_HEIGHT * DISTANCE_BETWEEN_CELLS / 2)
    if any(abs(c) > half for c, half in zip(charging_base_position, map_half_extents)):
        parser.error(f"--base-position {charging_base_position} is outside the map, "
                     f"+-{map_half_extents[0]:g} m x +-{map_half_extents[1]:g} m")

    ##### Module constant, read when the simulation is built #####
    main.ENCOUNTER_RADIUS = args.encounter_radius

    logs_directory = os.path.join(ROOT_DIR, args.logs_directory)
    os.makedirs(logs_directory, exist_ok=True)
    log_file = os.path.join(logs_directory, LOG_FILE)
    recording_file = os.path.join(logs_directory, RECORDING_FILE)

    mode = "train" if args.quiet else "test"
    if mode == "test":
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="w",
            format="%(message)s",
            force=True,
        )
        print(f"Protocol log: {log_file}")

    print(f"Running {args.drones} drones for {args.duration} simulated seconds ...")
    results, recorder, global_map, cost = run_simulation(
        duration=args.duration,
        number_of_drones=args.drones,
        mode=mode,
        monitor_interval=args.monitor_interval,
        sample_interval=args.sample_interval,
        enable_map_plot=args.map_plot,
        enable_simulation_plot=args.simulation_plot,
        individual=individual,
        charging_base_position=charging_base_position,
        seed=args.seed,
    )

    recorder.save(recording_file, results)
    print(f"Recording: {recording_file} ({len(recorder.times)} samples)")
    print(f"Plot it from the plots directory with: python3 plot_drones.py "
          f"--directory {args.logs_directory}  (same for plot_metrics.py, plot_more_metrics.py, "
          f"plot_states.py, plot_decisions.py, plot_estimation.py and plot_returns.py)")
    print()

    passed = report(results, recorder, global_map, cost, args.duration, args.drones,
                    monitor_interval=args.monitor_interval,
                    individual=individual,
                    charging_base_position=charging_base_position)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main_test())
