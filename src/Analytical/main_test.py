"""
Run it from the src directory, with gradysim on the PYTHONPATH:

    python3 -m Analytical.main_test
    python3 -m Analytical.main_test --duration 1000 --drones 2
    python3 -m Analytical.main_test --sample-interval 0.5   # finer global map sampling
    python3 -m Analytical.main_test --map-plot        # live per-drone map (very slow)
    python3 -m Analytical.main_test --simulation-plot # gradysim 3D view

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


SIMULATION_DURATION = 4000
NUMBER_OF_DRONES = 3

##### How often the drone state is sampled.
MONITOR_INTERVAL = 10.0

##### Cell spacing of the protocol, needed to turn metres into cells #####
DISTANCE_BETWEEN_CELLS = 20

DEFAULT_LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "main_test.log")
DEFAULT_RECORDING_FILE = os.path.join(os.path.dirname(__file__), "logs", "test_recording.pkl")


def make_recorded_drone(recorder_holder: dict, factory, **factory_kwargs):
    """
    Builds the configured protocol with the usual factory and wraps it so every
    change of destination is written to the recorder.

    The rest of the state is pulled from the simulator by the recorder itself.
    Destinations cannot be: they are decided between two samples, by three
    different paths (the drone's own choice, an encounter, a destination sent
    by another drone), so they are caught here instead, by watching
    goto_command after each event the protocol handles.
    """
    ConfiguredDrone = factory(**factory_kwargs)

    class RecordedDrone(ConfiguredDrone):
        def initialize(self) -> None:
            super().initialize()
            self._last_goto_command = np.array(self.goto_command, dtype=float)

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

        def handle_packet(self, message: str) -> None:
            super().handle_packet(message)
            self._record_decision()

    return RecordedDrone


def run_simulation(duration: int, number_of_drones: int,
                   mode: str = "test",
                   monitor_interval: float = MONITOR_INTERVAL,
                   sample_interval: float = main.GLOBAL_MAP_SAMPLE_INTERVAL,
                   enable_map_plot: bool = False,
                   enable_simulation_plot: bool = False):
    """
    Runs a single simulation of main.py, recording the state of every drone
    every monitor_interval seconds.

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
            BEST_INDIVIDUAL,
            mode=mode,
            enable_map_plot=enable_map_plot,
            enable_simulation_plot=enable_simulation_plot,
            sample_interval=sample_interval,
            observer_factory=make_recorder,
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
           monitor_interval: float = MONITOR_INTERVAL) -> bool:
    """
    Prints the summary of the run and returns whether every check passed.
    """
    total_cells = main.MAP_WIDTH * main.MAP_HEIGHT
    map_half_extent = main.MAP_WIDTH * DISTANCE_BETWEEN_CELLS / 2

    print("=== Scenario ===")
    print(f"Duration: {duration} s, drones: {number_of_drones}, "
          f"map: {main.MAP_WIDTH}x{main.MAP_HEIGHT} cells ({total_cells} cells)")
    print(f"Uncertainty rate: {main.UNCERTAINTY_RATE} every {main.VANISHING_UPDATE_TIME} s, "
          f"transmission range: {main.TRANSMISSION_RANGE} m")
    print(f"Recharge base at {main.CHARGING_BASE_POSITION}, map spans "
          f"+-{map_half_extent:.0f} m on each axis")
    print("Individual: " + ", ".join(
        f"{name}={value:g}" for (name, _, _), value in zip(GENE_BOUNDS, BEST_INDIVIDUAL)))
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

        visited = drone_results["unvisited_cells"]
        coverage = 100.0 * (total_cells - visited) / total_cells

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
    print(f"Encounters between drones: {Drone.Number_of_Encounters}")
    print(f"Global final uncertainty: {global_map.final_uncertainty:.1f} "
          f"(sum over the {total_cells} cells)")
    print(f"Global accumulated uncertainty: {global_map.accomulated_uncertainty:.1f} "
          f"(integrated every {global_map.sample_interval:g} s, {len(global_map.times)} samples)")
    print(f"Mean per-drone accumulated uncertainty: "
          f"{np.mean([r['accomulated_uncertainty'] for r in results.values()]):.1f}")
    print(f"Cost (the value the GA minimizes): {cost:.4f}")
    print()

    ##### Checks. Each one is a claim that has to hold for the run to be #####
    ##### considered healthy, printed so a failure is obvious.           #####
    print("=== Checks ===")
    checks = []

    mapping_drones = [drone_id for drone_id, drone_results in results.items()
                      if drone_results["unvisited_cells"] < total_cells]
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
        f"every drone stayed within {tolerance:.0f} m of the base"
        f"{'' if not far_drones else f' (outside: {far_drones})'}"))

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
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE,
                        help="where the per-routine protocol log is written "
                             "(default: %(default)s)")
    parser.add_argument("--recording-file", default=DEFAULT_RECORDING_FILE,
                        help="where the recording read by the plot scripts is written "
                             "(default: %(default)s)")
    parser.add_argument("--monitor-interval", type=float, default=MONITOR_INTERVAL,
                        help="simulated seconds between two samples of the drone state "
                             "(default: %(default)s)")
    parser.add_argument("--sample-interval", type=float,
                        default=main.GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map, "
                             "the integration step of the cost (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="run the protocol in train mode, without the per-routine log")
    parser.add_argument("--map-plot", action="store_true",
                        help="draw the live per-drone map (slow)")
    parser.add_argument("--simulation-plot", action="store_true",
                        help="open the gradysim visualization (slow)")
    args = parser.parse_args()

    mode = "train" if args.quiet else "test"
    if mode == "test":
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            filename=args.log_file,
            filemode="w",
            format="%(message)s",
            force=True,
        )
        print(f"Protocol log: {args.log_file}")

    print(f"Running {args.drones} drones for {args.duration} simulated seconds ...")
    results, recorder, global_map, cost = run_simulation(
        duration=args.duration,
        number_of_drones=args.drones,
        mode=mode,
        monitor_interval=args.monitor_interval,
        sample_interval=args.sample_interval,
        enable_map_plot=args.map_plot,
        enable_simulation_plot=args.simulation_plot,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.recording_file)), exist_ok=True)
    recorder.save(args.recording_file, results)
    print(f"Recording: {args.recording_file} ({len(recorder.times)} samples)")
    print(f"Plot it from the Analytical directory with: python3 logs/plot_drones.py "
          f"{args.recording_file}  (same for logs/plot_metrics.py and logs/plot_states.py)")
    print()

    passed = report(results, recorder, global_map, cost, args.duration, args.drones,
                    monitor_interval=args.monitor_interval)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main_test())
