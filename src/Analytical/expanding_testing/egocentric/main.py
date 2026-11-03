"""
Scenario of the multi-base tests: the egocentric system of Analytical/egocentric with
several recharge bases. It builds and scores one simulation, main_test.py is
the entry point. There is no GA here, the individuals come from the tuning of
Analytical/egocentric/main.py.

Differences with Analytical/egocentric/main.py:
- CHARGING_BASE_POSITIONS is a list, every base is a node of the simulation
  (recharge_base.py), added after the drones.
- The drones take off from the bases in turn, see protocol.take_off_base.
- The congestion is counted over all the bases together: Nc is the most drones
  charging at once, on any base, before the protocol and the cost react to it.
  The occupancy of each base is recorded too, for the report.
- The GA (training, objective function, base sampling) is not here.
"""

import random
import logging

from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .protocol import drone_protocol_factory, DroneStatus, take_off_base
from .recharge_base import ChargingRegistry, recharge_base_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from gradysim.simulator.handler.mobility import (
    DynamicVelocityMobilityConfiguration,
    DynamicVelocityMobilityHandler,
)

import numpy as np

how_many_simulations = 0

##### Scenario. main_test.py overrides MAP_WIDTH/MAP_HEIGHT and TRANSMISSION_RANGE from its command line
SIMULATION_DURATION = 10000
MAP_WIDTH = 100   # cells, 20 m apart: 2000 m x 2000 m centred on (0, 0). The protocol needs a square map
MAP_HEIGHT = 100
NUMBER_OF_DRONES = 3
UNCERTAINTY_RATE = 0.001
VANISHING_UPDATE_TIME = 1.0
TRANSMISSION_RANGE = 200
DRONE_SPEED = 15.0  # cruise speed, m/s. Also the speed limit of the mobility handler
DRONE_MAX_ACCELERATION = 3.0  # horizontal acceleration limit of the mobility handler, m/s^2

##### Fixed protocol parameters, not tuned
DISCHARGE_RATE = 0.00073  # measured: 0.000725-0.000733 per second, flying at 15 m/s with the random air speed (0.00077 at 10 m/s)
CHARGING_BASE_POSITIONS = [(0.0, 0.0)]  # [(x, y), ...] in metres, one per base
DISTANCE_BETWEEN_CELLS = 20  # cell spacing, m. Same as the protocol's, needed to put the base on a cell
KERNEL_N_SIGMA = 3  # kernel truncation radius, in sigmas. Above ~3 the cut-off weight is negligible
INITIAL_UNCERTAINTY = 0.5  # uncertainty of every cell at t=0, the first sweep of the map weighs less in the cost
MIN_CHARGE_MARGIN = 0.20  # fraction of battery for going near the base of recharging
##### The egocentric bases send nothing, they are placed exactly as in the swarm system so both run the same nodes. #####
##### There the range of gradysim is 3D, and the antenna is at the drones' DRONE_ALTITUDE so the base heartbeats    #####
##### reach as far as the drone-to-drone messages.                                                                  #####
BASE_ANTENNA_ALTITUDE = 50.0

##### Cost weights, every term is integrated over time.
##### UNCERTAINTY_WEIGHT was 0.01 on the 50x50 map at 10 m/s. On 100x100 at 15 m/s the accumulated #####
##### uncertainty of BEST_INDIVIDUAL is 13.9x larger (GA aggregate, 3 seeds x sizes 3/5/7/10), so the  #####
##### weight is divided by it: the uncertainty term keeps its old size next to the penalties.         #####
UNCERTAINTY_WEIGHT = 0.00072          # cost per unit of global uncertainty per simulated second
DEAD_DRONE_PENALTY = 1000.0           # cost per dead drone per simulated second
CHARGING_CONGESTION_PENALTY = 30.0    # K of congestion_penalty_rate, cost per simulated second
CHARGING_CONGESTION_STEEPNESS = 5.0   # beta of congestion_penalty_rate, how fast the cost grows with the share of the swarm above Nc
COST_TERMS = ("uncertainty", "dead", "charging")  # the terms summed into the cost, see simulation_cost_terms

##### How often the global map is assembled and sampled, in simulated seconds.
GLOBAL_MAP_SAMPLE_INTERVAL = 1.0

##### Genome layout of Analytical/egocentric/main.py: (name, low, high), in the order the individual stores them.
GENE_BOUNDS = [
    ("base_variance",            0.25,  100.0),   # sigma_0^2 of Eq. (11), in cells^2
    ("alpha_variance_modifier",  0.0,   100.0),   # alpha of Eq. (11)
    ("energy_gamma",             0.10,  100.0),   # gamma of Eqs. (14)/(15)
    ("charging_base_multiplier", 0.0,  100.0),   # kappa of Eq. (15)
    ("distance_between_drone_norm", 10.0, 1000.0),
    ("charge_margin", MIN_CHARGE_MARGIN, 0.5),
]

##### Default individual, the BEST_INDIVIDUAL of Analytical/egocentric/main.py. main_test.py takes another one #####
##### with --individual, e.g. the "Parameters:" of a logbook in Analytical/egocentric/ga_logs.                  #####
##### Tuned on the old scenario (50x50 map, 10 m/s, base fixed at the map centre) #####
BEST_INDIVIDUAL = [76.31, 40.14, 7.44, 52.38, 455.63, 0.26]


def unpack_individual(individual):
    """
    Maps the flat GA genome onto the protocol's tunable parameters, in the
    order declared by GENE_BOUNDS.
    """
    base_variance, alpha, gamma, kappa, drone_norm, charge_margin = individual
    return dict(
        base_variance=base_variance,
        alpha_variance_modifier=alpha,
        energy_gamma=gamma,
        charging_base_multiplier=kappa,
        distance_between_drone_norm=drone_norm,
        chage_margin=charge_margin

    )


def cell_centre(position) -> tuple:
    """
    (x, y) of the centre of the cell position is in. A drone going to a base flies to the centre of
    the base's cell (FitnessEvaluator.nearest_base_cell), so a base placed there is the point the drones
    take off from, charge at, and the bases send their heartbeats from.
    """
    map_center_offset = MAP_WIDTH * DISTANCE_BETWEEN_CELLS / 2
    cell = np.clip(np.round((np.asarray(position, dtype=float) + map_center_offset) / DISTANCE_BETWEEN_CELLS),
                   0, [MAP_WIDTH - 1, MAP_HEIGHT - 1])
    return tuple(float(c) for c in cell * DISTANCE_BETWEEN_CELLS - map_center_offset)


def default_charging_capacity(number_of_drones: int) -> int:
    """
    Nc when none is given: the capacity of the single base of Analytical/egocentric, now over all the
    bases together. At least number_of_drones - Nc drones are meant to be mapping at any time.
    """
    return int(number_of_drones/3)


def congestion_penalty_rate(excess: int, number_of_drones: int) -> float:
    """
    Cost per simulated second of having excess drones CHARGING above the
    capacity: K * (exp(beta * excess / number_of_drones) - 1). It is zero with
    no excess, grows faster than linearly with every extra drone, and the same
    excess costs more in a smaller swarm, where it is a bigger share of it.
    """
    return CHARGING_CONGESTION_PENALTY * (
        np.exp(CHARGING_CONGESTION_STEEPNESS * excess / number_of_drones) - 1.0)


class GlobalMapMonitor:
    """
    Assembles and samples the global map of the swarm while the simulation runs.

    The global map is the element-wise minimum of the uncertainty maps of every
    drone: a cell is as well known as the best informed drone believes it to be.
    No drone ever holds this map, each one only knows its own observations plus
    whatever it was told during an encounter, so it cannot be read from a
    protocol at the end of the run. It only exists as a property of the system,
    and it has to be assembled from the outside while the simulation is still
    running.

    accomulated_uncertainty is the integral of the global uncertainty over time,
    in uncertainty*second. That is the real cost of the simulation, the value the
    GA minimizes.

    The status of the swarm is sampled at the same instants, for the penalties:
    accomulated_dead_time is the integral of the number of DEAD drones, in
    drone*second, and accomulated_charging_excess the integral of the number of
    drones CHARGING above nc_charging_base, over all the bases together, also in
    drone*second. accomulated_charging_penalty is the integral of
    congestion_penalty_rate of that number, the congestion term of the cost. It
    is not linear, so it has to be accumulated sample by sample.

    charging_drones_per_base keeps how many drones charge on each base at every
    sample. It does not enter the cost, it is there to know where the drones
    charge.
    """

    def __init__(self, simulation, number_of_drones: int, sample_interval: float,
                 number_of_bases: int = 1, nc_charging_base: int = None):
        self.sim = simulation
        self.number_of_drones = number_of_drones
        self.sample_interval = sample_interval
        self.number_of_bases = number_of_bases
        self.nc_charging_base = default_charging_capacity(number_of_drones) if nc_charging_base is None else nc_charging_base
        self._protocols = None

        self.times = []
        self.uncertainty = []
        self.unvisited = []
        self.accomulated_uncertainty = 0.0
        self.map = None
        self.visited = None

        self.dead_drones = []
        self.charging_drones = []
        self.charging_drones_per_base = []  # one array of number_of_bases counts per sample
        self.accomulated_dead_time = 0.0
        self.accomulated_charging_excess = 0.0
        self.accomulated_charging_penalty = 0.0

    def sample(self, time: float) -> None:
        if self._protocols is None:
            ##### The protocols only exist once the simulation has been
            ##### initialized, which happens on its first step. The drones are
            ##### the first nodes, the bases come after them.
            self._protocols = [self.sim.get_node(i).protocol_encapsulator.protocol
                               for i in range(self.number_of_drones)]

        self.map = np.min([p.map[:, :, 0] for p in self._protocols], axis=0)
        self.visited = np.any([p.is_cell_visited > 0 for p in self._protocols], axis=0)

        uncertainty = float(self.map.sum())
        self.accomulated_uncertainty += uncertainty * self.sample_interval

        self.times.append(float(time))
        self.uncertainty.append(uncertainty)
        self.unvisited.append(int(self.visited.size - self.visited.sum()))

        ##### Penalties. A dead drone counts for every sample until the end of #####
        ##### the run, so the earlier it dies the more it costs.                #####
        dead = sum(p.status == DroneStatus.DEAD for p in self._protocols)
        charging_per_base = np.bincount([p.recharge_base_index for p in self._protocols
                                         if p.status == DroneStatus.CHARGING],
                                        minlength=self.number_of_bases)
        charging = int(charging_per_base.sum())
        self.accomulated_dead_time += dead * self.sample_interval
        excess = max(0, charging - self.nc_charging_base)
        self.accomulated_charging_excess += excess * self.sample_interval
        self.accomulated_charging_penalty += (congestion_penalty_rate(excess, self.number_of_drones)
                                              * self.sample_interval)

        self.dead_drones.append(dead)
        self.charging_drones.append(charging)
        self.charging_drones_per_base.append(charging_per_base)

    @property
    def final_uncertainty(self) -> float:
        return self.uncertainty[-1] if self.uncertainty else float("nan")

    @property
    def unvisited_cells(self) -> int:
        return self.unvisited[-1] if self.unvisited else 0


def run_stepped_simulation(simulation, sample_interval: float, observers=()) -> None:
    """
    Runs the simulation one event at a time, calling every observer with the
    current simulated time once per sample_interval.

    step_simulation() is used instead of start_simulation() because the swarm
    level quantities, the global map above all, are not held by any protocol
    and have to be read from the outside, while the run is still going.
    """
    def notify(time):
        for observe in observers:
            observe(time)

    ##### The first step initializes every protocol, only after it the
    ##### drones have a map to sample.
    running = simulation.step_simulation()
    notify(simulation._current_timestamp)
    if not running:
        return

    next_sample = sample_interval
    while running:
        while running and simulation._current_timestamp < next_sample:
            running = simulation.step_simulation()
        if not running:
            break
        while next_sample <= simulation._current_timestamp:
            notify(next_sample)
            next_sample += sample_interval

    ##### The loop above stops on the event that ends the simulation, so the
    ##### final state would otherwise be missing from every observer.
    notify(simulation._current_timestamp)


#### GradySim function
def create_and_run_simulation(individual, mode: str = "train",
                              enable_map_plot: bool = False,
                              enable_simulation_plot: bool = False,
                              sample_interval: float = GLOBAL_MAP_SAMPLE_INTERVAL,
                              observer_factory=None,
                              number_of_drones: int = None,
                              seed: int = None,
                              charging_base_positions: list = None,
                              charging_capacity: int = None):
    """
    Runs one simulation with the given individual and returns
    (results_aggregator, global_map): the per drone results filled in by
    finish() and the GlobalMapMonitor sampled during the run. The behavior is
    the same for both modes, the only difference is that "test" writes the
    detailed logs, based on the recorder.py, and may draw the plots.

    The simulation is driven event by event so the global map can be sampled
    every sample_interval seconds, see run_stepped_simulation.

    observer_factory, when given, is called with the built simulation and must
    return a callable observe(time), invoked on every sample next to the global
    map monitor. main_test.py uses it to record the run.

    number_of_drones is the size of the swarm. When None, NUMBER_OF_DRONES is
    read at call time, so main_test.py can still override the module value.

    seed, when given, fixes the scenario: every drone draws its initial
    destination and its wind from its own streams, seeded by (seed, drone id),
    see the protocol's initialize. When None the run is random.

    charging_base_positions is the list of the (x, y) of the recharge bases,
    each one should be on a cell centre (see cell_centre). The drones take off
    from them in turn. When None, CHARGING_BASE_POSITIONS is used.

    charging_capacity is Nc, the most drones charging at once, over all the
    bases together, before the congestion terms start, in the protocol and in
    the cost. When None, default_charging_capacity.
    """
    ##### Configuring global parameter
    global how_many_simulations
    how_many_simulations += 1

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if number_of_drones is None:
        number_of_drones = NUMBER_OF_DRONES

    if charging_base_positions is None:
        charging_base_positions = CHARGING_BASE_POSITIONS
    charging_base_positions = [(float(x), float(y)) for x, y in charging_base_positions]

    if charging_capacity is None:
        charging_capacity = default_charging_capacity(number_of_drones)

    ##### Configuring the simulation
    config = SimulationConfiguration(
        duration=SIMULATION_DURATION,
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    # BatteryPowerPlugin requires DynamicVelocityMobilityHandler for velocity telemetry.
    mobility_config = DynamicVelocityMobilityConfiguration(
        update_rate=0.05,
        max_speed_xy=DRONE_SPEED,
        max_speed_z=10.0,
        max_acc_xy=DRONE_MAX_ACCELERATION,
        max_acc_z=3.0,
        send_telemetry=True,
    )
    builder.add_handler(DynamicVelocityMobilityHandler(mobility_config))
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=TRANSMISSION_RANGE
    )))

    if enable_simulation_plot:
        builder.add_handler(VisualizationHandler())

    ##### Shared by the bases and the drones of this simulation only, see recharge_base.py #####
    charging_registry = ChargingRegistry(len(charging_base_positions))

    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=UNCERTAINTY_RATE,
        vanishing_update_time=VANISHING_UPDATE_TIME,
        number_of_drones=number_of_drones,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        drone_speed=DRONE_SPEED,
        drone_max_acceleration=DRONE_MAX_ACCELERATION,
        **unpack_individual(individual),
        discharge_rate=DISCHARGE_RATE,
        charging_base_positions=charging_base_positions,
        charging_registry=charging_registry,
        min_charge_margin=MIN_CHARGE_MARGIN,
        Nc_charging_base=charging_capacity,
        results_aggregator=results_aggregator,
        mode=mode,
        enable_map_plot=enable_map_plot,
        kernel_n_sigma=KERNEL_N_SIGMA,
        initial_uncertainty=INITIAL_UNCERTAINTY,
        seed=seed
    )

    ##### The drones take off from the recharge bases, in turn #####
    for drone_id in range(number_of_drones):
        x, y = charging_base_positions[take_off_base(drone_id, len(charging_base_positions))]
        builder.add_node(ConfiguredDrone, (x, y, 0))

    ##### The bases are added after the drones, so the drones keep the ids 0..number_of_drones-1 #####
    ##### the protocol indexes its swarm arrays with. Base b is node number_of_drones + b.         #####
    for base_index, (x, y) in enumerate(charging_base_positions):
        builder.add_node(recharge_base_factory(base_index, charging_registry, mode=mode),
                         (x, y, BASE_ANTENNA_ALTITUDE))

    # Building & starting
    simulation = builder.build()

    global_map = GlobalMapMonitor(simulation, number_of_drones, sample_interval,
                                  number_of_bases=len(charging_base_positions),
                                  nc_charging_base=charging_capacity)
    observers = [global_map.sample]
    if observer_factory is not None:
        observers.append(observer_factory(simulation))

    run_stepped_simulation(simulation, sample_interval, observers)

    return results_aggregator, global_map


def simulation_cost_terms(global_map: GlobalMapMonitor) -> dict:
    """
    The weighted terms of the cost of one simulation, keyed by COST_TERMS. Their
    sum is the value minimized, they are kept apart so it can be seen which one
    drives it.
    """
    return dict(
        uncertainty=UNCERTAINTY_WEIGHT * global_map.accomulated_uncertainty,
        dead=DEAD_DRONE_PENALTY * global_map.accomulated_dead_time,
        charging=global_map.accomulated_charging_penalty,
    )


def format_cost_terms(terms: dict) -> str:
    return ", ".join(f"{term}: {terms[term]:.6g}" for term in COST_TERMS)


def evaluate_simulation_cost(results_aggregator, global_map: GlobalMapMonitor,
                             mode: str = "train"):
    """
    Turns the result of the simulation into the single value to be minimized.
    """
    ##### Getting the results of the simulation #####
    number_of_drones = global_map.number_of_drones
    medium_uncertainty = 0
    for i in range(number_of_drones):
        medium_uncertainty += results_aggregator[i]['accomulated_uncertainty']/number_of_drones

    ##### Penalties, accumulated by the monitor on every sample #####
    terms = simulation_cost_terms(global_map)
    dead_penalty = terms["dead"]
    charging_penalty = terms["charging"]

    ##### Cost for optimization #####
    total_cost = sum(terms.values())

    if mode == "test":
        logging.info(f"Global accomulated uncertainty: {global_map.accomulated_uncertainty}, cost: {terms['uncertainty']}")
        logging.info(f"Global final uncertainty: {global_map.final_uncertainty}")
        logging.info(f"Global unvisited cells: {global_map.unvisited_cells}")
        logging.info(f"Medium accomulated uncertainty per drone: {medium_uncertainty}")
        logging.info(f"Dead drone time: {global_map.accomulated_dead_time} drone*s, penalty: {dead_penalty}")
        logging.info(f"Charging above Nc, over all the bases: {global_map.accomulated_charging_excess} drone*s, penalty: {charging_penalty}")
        logging.info(f"Cost: {total_cost}")

    return total_cost
