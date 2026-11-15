"""
Single test run of the MARL system, the counterpart of Analytical/swarm/main_test.py:
same scenario, same report, same checks, same recording and plots.

Run it from this directory, with gradysim on the PYTHONPATH:

    python3 main_test.py --checkpoint checkpoints_energy/best.pt          # the trained network
    python3 main_test.py --checkpoint checkpoints_energy/best.pt --drones 5 --duration 4000 --seed 1
    python3 main_test.py --policy analytical          # the analytical protocol, through this environment
    python3 main_test.py --policy analytical --individual 90.6 95.4 ...   # a tuned individual
    python3 main_test.py --policy random              # random valid actions, a lower bound
    python3 main_test.py --transmission-range 500 --base-position 400 -200
    python3 main_test.py --logs_directory logs_3     # keep this run apart
    python3 main_test.py --map-plot                  # live per-drone map (very slow)
    python3 main_test.py --simulation-plot           # gradysim 3D view

--policy nn (the default) runs the network of --checkpoint greedily (no exploration).
--policy analytical takes every decision with the fitness function of Analytical/swarm:
with the same --seed, --individual, base and range the run is the one of
Analytical/swarm/main_test.py (check_equivalence.py checks it), so it is the
baseline to compare the network with, on the same scenario.

The protocol log and the recording go to the logs directory, next to this file
(logs by default). Plot it from the plots directory with:

    python3 plot_drones.py --directory logs_3

The exit status is 0 when every check passes and 1 otherwise.
"""

import os
import sys
import json
import logging
import argparse

import numpy as np

import config
from env.mapping_environment import MappingEnvironment
from env.gradysim_environment.protocol import (DroneStatus, Drone, MessageType, DecisionKind,
                                               drone_protocol_factory)
from simulation_orchestrator import AsyncMARLOrchestrator
from policy import RandomPolicy, load_greedy_policy
from recorder import SimulationRecorder
from estimation_metrics import summarize, format_summary


SIMULATION_DURATION = config.SIMULATION_DURATION
NUMBER_OF_DRONES = 10
POLICIES = ("nn", "analytical", "random")
DEFAULT_CHECKPOINT = os.path.join("checkpoints_energy", "best.pt")  # CHECKPOINT_DIR of train.py

##### How often the drone state is sampled.
MONITOR_INTERVAL = 10.0

##### Cell spacing of the protocol, needed to turn metres into cells #####
DISTANCE_BETWEEN_CELLS = config.DISTANCE_BETWEEN_CELLS

##### Each run is written into a logs directory next to this file, so runs can #####
##### be kept apart. The file names are fixed, plots/*.py look them up there.   #####
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGS_DIRECTORY = "logs"
LOG_FILE = "main_test.log"
RECORDING_FILE = "test_recording.pkl"

##### What made a drone choose the base, abort or die, as in Analytical/swarm/main_test.py: the event #####
##### the protocol was handling when its status changed, or what made it ask for the decision.      #####
TIMER_TRIGGERS = {"mobility": "own", "recharge_done": "after_recharge", "battery_check": "battery"}
MESSAGE_TRIGGERS = {"GOSSIPING_MESSAGE": "encounter",
                    "SHARE_GOTO_POSITION_MESSAGE": "sent",
                    "CHARGING_NOTIFICATION_MESSAGE": "charging_notification"}
DECISION_TRIGGERS = {"arrival": "own", "recharge_done": "after_recharge",
                     "charging_notification": "charging_notification", "encounter": "encounter"}


def message_trigger(message: str) -> tuple:
    """(trigger, sender) of a packet, see MESSAGE_TRIGGERS."""
    data = json.loads(message)
    try:
        name = MessageType(data.get("message_type")).name
    except ValueError:
        name = str(data.get("message_type"))
    return MESSAGE_TRIGGERS.get(name, name.lower()), data.get("sender")


def make_recorded_drone(recorder_holder: dict, factory, **factory_kwargs):
    """
    Builds the configured protocol with the usual factory and wraps it so every
    change of destination and every trip to the base is written to the recorder,
    as in Analytical/swarm/main_test.py.

    Here a decision is applied after the event that asked for it (apply_decision), so
    while the drone has a decision pending nothing is recorded: the change is
    recorded once the decision is applied, with what made the drone ask for it, and
    the run is recorded as the analytical one, where the decision is taken inside the
    event.
    """
    ConfiguredDrone = factory(**factory_kwargs)

    class RecordedDrone(ConfiguredDrone):
        def initialize(self) -> None:
            super().initialize()
            self._last_goto_command = np.array(self.goto_command, dtype=float)
            self._last_status = self.status

        def _deciding(self) -> bool:
            return any(request.decider_id == self.provider.get_id() for request in self.decision_requests)

        def _record_return(self, trigger_of) -> None:
            """
            Records the drone choosing the base, reaching it, being sent back to
            mapping on the way, or dying, with the battery it has at that moment.
            """
            previous, current = self._last_status, self.status
            if previous == current:
                return
            self._last_status = current

            if current == DroneStatus.GOING_TO_BASE:
                event = "decision"
            elif current == DroneStatus.DEAD:
                event = "death"
            elif previous == DroneStatus.GOING_TO_BASE and current == DroneStatus.CHARGING:
                event = "arrival"
            elif previous == DroneStatus.GOING_TO_BASE and current == DroneStatus.MAPPING:
                event = "abort"
            else:
                ##### CHARGING -> MAPPING, the end of the recharge, is not part of a trip #####
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
            if self._deciding():
                return
            self._record_decision()
            self._record_return(lambda: (TIMER_TRIGGERS.get(timer, timer), None))

        def handle_packet(self, message: str) -> None:
            super().handle_packet(message)
            if self._deciding():
                return
            self._record_decision()
            self._record_return(lambda: message_trigger(message))

        def apply_decision(self, request, action) -> None:
            super().apply_decision(request, action)
            if self._deciding():
                return
            self._record_decision()
            self._record_return(lambda: (DECISION_TRIGGERS.get(request.trigger, request.trigger),
                                         request.trigger_sender))

    return RecordedDrone


def make_policy(policy: str, checkpoint: str):
    if policy == "nn":
        import torch
        if torch.load(checkpoint, map_location="cpu").get("actor_kwargs") is None:
            ##### A checkpoint without the network arguments: the ones of train.py now #####
            import train
            return load_greedy_policy(checkpoint, **train.actor_kwargs())
        return load_greedy_policy(checkpoint)
    ##### With the analytical backend the environment decides, nothing is ever flagged #####
    return RandomPolicy()


def run_simulation(duration: int, number_of_drones: int,
                   policy: str = "nn",
                   checkpoint: str = DEFAULT_CHECKPOINT,
                   mode: str = "test",
                   monitor_interval: float = MONITOR_INTERVAL,
                   sample_interval: float = config.GLOBAL_MAP_SAMPLE_INTERVAL,
                   enable_map_plot: bool = False,
                   enable_simulation_plot: bool = False,
                   individual=config.BEST_INDIVIDUAL,
                   charging_base_position: tuple = config.CHARGING_BASE_POSITION,
                   transmission_range: float = config.TRANSMISSION_RANGE,
                   seed: int = None,
                   record: bool = True):
    """
    Runs a single simulation with the given policy and recharge base, recording
    the state of every drone every monitor_interval seconds (when record is set).
    seed fixes the wind and the initial destinations, None is a random run.

    Returns (results, recorder, monitor, env): the per drone results filled in by
    finish(), the SimulationRecorder (None without record), the SwarmMonitor of the
    run (the global map, the cost) and the closed environment (decisions, reward).
    """
    recorder_holder = {}
    ##### Time of the last sample offered and of the last one kept,
    ##### so the end of the run can be recorded without duplicating a sample.
    last_observed = [None]
    last_recorded = [None]

    def recorded_factory(**factory_kwargs):
        return make_recorded_drone(recorder_holder, drone_protocol_factory, **factory_kwargs)

    def make_recorder(simulation):
        recorder = SimulationRecorder(
            simulation,
            number_of_drones,
            config.MAP_WIDTH,
            config.MAP_HEIGHT,
            distance_between_cells=DISTANCE_BETWEEN_CELLS,
            status_labels={status.value: status.name for status in DroneStatus},
        )
        recorder_holder["recorder"] = recorder

        ##### Decimation of the samples, so the recording stays at monitor_interval. #####
        ##### A full map per drone is stored on each one.                            #####
        next_sample = [0.0]

        def observe(time):
            last_observed[0] = time
            if time < next_sample[0]:
                return
            recorder.snapshot(time)
            last_recorded[0] = time
            ##### Anchored on the grid, not on the time of the sample taken #####
            while next_sample[0] <= time:
                next_sample[0] += monitor_interval

        return observe

    env = MappingEnvironment(config.scenario_config(
        render_mode="visual" if enable_simulation_plot else None,
        sample_interval=sample_interval,
        num_agents_choices=(number_of_drones,),
        seed=seed,
        charging_base_position=tuple(charging_base_position),
        max_episode_length=duration,
        communication_range=transmission_range,
        end_when_all_dead=False,
        decision_backend="analytical" if policy == "analytical" else "policy",
        analytical_parameters=config.unpack_individual(individual) if policy == "analytical" else None,
        mode=mode,
        enable_map_plot=enable_map_plot,
        protocol_factory=recorded_factory if record else None,
        observer_factory=make_recorder if record else None,
    ))
    orchestrator = AsyncMARLOrchestrator(env, make_policy(policy, checkpoint), collect_transitions=False)

    td = orchestrator.reset()
    while True:
        td = orchestrator.step(td)
        if td["next", "done"].item():
            break
        td = td["next"]
    env.close()

    ##### The run ends on the event that closes the simulation, which can fall
    ##### before the next multiple of monitor_interval, so the final state is
    ##### added here rather than being left out of the recording.
    recorder = recorder_holder.get("recorder")
    if recorder is not None and last_observed[0] is not None and last_observed[0] != last_recorded[0]:
        recorder.snapshot(last_observed[0])

    return env.results_aggregator, recorder, env.monitor, env


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


def describe_policy(policy: str, checkpoint: str, individual) -> str:
    if policy == "analytical":
        return "analytical fitness function, individual: " + ", ".join(
            f"{name}={value:g}" for name, value in zip(config.GENE_NAMES, individual))
    if policy == "random":
        return "random valid actions"
    return f"neural network {checkpoint}"


def report(results: dict, recorder: SimulationRecorder, monitor, env,
           duration: int, number_of_drones: int,
           policy_description: str,
           monitor_interval: float = MONITOR_INTERVAL,
           charging_base_position: tuple = config.CHARGING_BASE_POSITION,
           transmission_range: float = config.TRANSMISSION_RANGE) -> bool:
    """
    Prints the summary of the run and returns whether every check passed. The same
    report as Analytical/swarm/main_test.py, plus the decisions and the reward.
    """
    total_cells = config.MAP_WIDTH * config.MAP_HEIGHT
    map_half_extent = config.MAP_WIDTH * DISTANCE_BETWEEN_CELLS / 2

    print("=== Scenario ===")
    print(f"Duration: {duration} s, drones: {number_of_drones}, "
          f"map: {config.MAP_WIDTH}x{config.MAP_HEIGHT} cells ({total_cells} cells)")
    print(f"Uncertainty rate: {config.UNCERTAINTY_RATE} every {config.VANISHING_UPDATE_TIME} s, "
          f"transmission range: {transmission_range} m, speed: {config.DRONE_SPEED} m/s")
    print(f"Recharge base at {tuple(charging_base_position)}, map spans "
          f"+-{map_half_extent:.0f} m on each axis")
    print(f"Policy: {policy_description}")
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
    ##### what the system knows as a whole.                                       #####
    print("=== Swarm (global map) ===")
    swarm_visited = total_cells - monitor.unvisited_cells
    print(f"Cells seen by at least one drone: {swarm_visited}/{total_cells} "
          f"({100.0 * swarm_visited / total_cells:.1f}%)")
    print(f"Encounters between drones: {Drone.Number_of_Encounters}")
    print(f"Global final uncertainty: {monitor.final_uncertainty:.1f} "
          f"(sum over the {total_cells} cells)")
    print(f"Global accumulated uncertainty: {monitor.accomulated_uncertainty:.1f} "
          f"(integrated every {monitor.sample_interval:g} s, {len(monitor.times)} samples)")
    print(f"Mean per-drone accumulated uncertainty: "
          f"{np.mean([r['accomulated_uncertainty'] for r in results.values()]):.1f}")
    charging = np.array(monitor.charging_drones)
    print(f"Drones charging: at most {int(charging.max())} at once, more than Nc = {monitor.nc_charging_base} "
          f"{100.0 * np.mean(charging > monitor.nc_charging_base):.1f}% of the time")
    terms = monitor.cost_terms()
    print(f"Cost of Analytical/swarm (the value its GA minimizes): {monitor.cost():.4f} "
          f"({', '.join(f'{name}: {value:.6g}' for name, value in terms.items())})")
    print()

    ##### What the agents decided and the reward they got #####
    print("=== Decisions and reward ===")
    kinds = ", ".join(f"{kind.lower()} {count}" for kind, count in env.decisions.items())
    print(f"Decisions: {sum(env.decisions.values())} ({kinds})")
    metrics = env.episode_metrics()
    print("Reward of the swarm: {:+.2f} (uncertainty {:+.2f}, congestion {:+.2f}, {} deaths)".format(
        metrics["reward"], metrics["reward_global_uncertainty"], -metrics["reward_congestion"],
        int(metrics["reward_deaths"])))
    print()

    ##### How wrong the drones are about each other's battery, and at which battery #####
    ##### they go back. plots/plot_estimation.py and plot_returns.py draw the same. #####
    print("=== Battery estimation and returns to the base ===")
    status_values = {status.name: status.value for status in DroneStatus}
    print(format_summary(summarize(recorder.drone_arrays(), recorder.returns, status_values)))
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
    best_drone_uncertainty = min(r["final_uncertainty"] for r in results.values())
    checks.append((
        monitor.final_uncertainty <= best_drone_uncertainty + 1e-6,
        f"the global map ({monitor.final_uncertainty:.1f}) is at least as good as "
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

    for passed, description in checks:
        print(f"[{'ok  ' if passed else 'FAIL'}] {description}")

    return all(passed for passed, _ in checks)


def main_test() -> int:
    parser = argparse.ArgumentParser(
        description="Single simulation of the MARL system, to check that the "
                    "drones map the area and go back to the recharge base")
    parser.add_argument("--policy", choices=POLICIES, default="nn",
                        help="who decides: the network of --checkpoint, the analytical fitness "
                             "function, or random valid actions (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="checkpoint of the network, relative to this directory "
                             "(default: %(default)s)")
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
    parser.add_argument("--sample-interval", type=float, default=config.GLOBAL_MAP_SAMPLE_INTERVAL,
                        help="simulated seconds between two samples of the global map, "
                             "the integration step of the cost (default: %(default)s)")
    parser.add_argument("--individual", type=float, nargs=len(config.GENE_NAMES), default=config.BEST_INDIVIDUAL,
                        metavar=tuple(name.upper() for name in config.GENE_NAMES),
                        help="--policy analytical: genes of the individual, in the GENE_BOUNDS order of "
                             "Analytical/swarm/main.py (default: BEST_INDIVIDUAL, %(default)s)")
    parser.add_argument("--transmission-range", type=float, default=config.TRANSMISSION_RANGE,
                        help="communication range of the drones, in m (default: %(default)s)")
    parser.add_argument("--base-position", type=float, nargs=2, default=config.CHARGING_BASE_POSITION,
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

    charging_base_position = tuple(args.base_position)
    map_half_extents = (config.MAP_WIDTH * DISTANCE_BETWEEN_CELLS / 2,
                        config.MAP_HEIGHT * DISTANCE_BETWEEN_CELLS / 2)
    if any(abs(c) > half for c, half in zip(charging_base_position, map_half_extents)):
        parser.error(f"--base-position {charging_base_position} is outside the map, "
                     f"+-{map_half_extents[0]:g} m x +-{map_half_extents[1]:g} m")
    checkpoint = os.path.join(ROOT_DIR, args.checkpoint)
    if args.policy == "nn" and not os.path.exists(checkpoint):
        parser.error(f"no checkpoint {checkpoint}: train the network first (train.py), give another "
                     f"--checkpoint, or run --policy analytical / random")

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
    results, recorder, monitor, env = run_simulation(
        duration=args.duration,
        number_of_drones=args.drones,
        policy=args.policy,
        checkpoint=checkpoint,
        mode=mode,
        monitor_interval=args.monitor_interval,
        sample_interval=args.sample_interval,
        enable_map_plot=args.map_plot,
        enable_simulation_plot=args.simulation_plot,
        individual=args.individual,
        charging_base_position=charging_base_position,
        transmission_range=args.transmission_range,
        seed=args.seed,
    )

    recorder.save(recording_file, results)
    print(f"Recording: {recording_file} ({len(recorder.times)} samples)")
    print(f"Plot it from the plots directory with: python3 plot_drones.py "
          f"--directory {args.logs_directory}  (same for plot_metrics.py, plot_more_metrics.py, "
          f"plot_states.py, plot_decisions.py, plot_estimation.py and plot_returns.py)")
    print()

    passed = report(results, recorder, monitor, env, args.duration, args.drones,
                    describe_policy(args.policy, args.checkpoint, args.individual),
                    monitor_interval=args.monitor_interval,
                    charging_base_position=charging_base_position,
                    transmission_range=args.transmission_range)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main_test())
