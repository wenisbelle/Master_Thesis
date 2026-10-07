import os
import random
import logging
import argparse
import multiprocessing

from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .protocol import drone_protocol_factory, DroneStatus
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from gradysim.simulator.handler.mobility import (
    DynamicVelocityMobilityConfiguration,
    DynamicVelocityMobilityHandler,
)

from deap import algorithms, base, creator, tools
import numpy as np

how_many_simulations = 0
CORES_TO_USE = 16

##### Scenario, shared by the tuning and the test runs 
SIMULATION_DURATION = 10000
MAP_WIDTH = 50
MAP_HEIGHT = 50
NUMBER_OF_DRONES = 5  # used by the test runs, the GA tuning uses TRAINING_DRONE_COUNTS
UNCERTAINTY_RATE = 0.001
VANISHING_UPDATE_TIME = 1.0
TRANSMISSION_RANGE = 200

##### Fixed protocol parameters, not tuned 
DISCHARGE_RATE = 0.00075  # measured: 0.00074-0.00078 per second, flying at 10 m/s with the random air speed
CHARGING_BASE_POSITION = (0.0, 0.0)
MIN_CHARGE_MARGIN = 0.20  # fraction of battery for going near the base of recharging
NC_CHARGING_BASE = 2  # number of drones that can charge at the same time in the base. A bigger number will start receiving penalty

##### Cost penalties, both integrated over time like the uncertainty.
DEAD_DRONE_PENALTY = 10000.0           # cost per dead drone per simulated second
CHARGING_CONGESTION_PENALTY = 100.0  # cost per drone above NC_CHARGING_BASE per simulated second

##### How often the global map is assembled and sampled, in simulated seconds. 
GLOBAL_MAP_SAMPLE_INTERVAL = 1.0

##### GA genome layout: (name, low, high), in the order the individual stores them.
##### These bounds are PLACEHOLDERS chosen to be dimensionally sane, they have not
##### been calibrated against the scenario yet.
GENE_BOUNDS = [
    ("base_variance",            0.25,  100.0),   # sigma_0^2 of Eq. (11), in cells^2
    ("alpha_variance_modifier",  0.0,   100.0),   # alpha of Eq. (11)
    ("energy_gamma",             0.10,  100.0),   # gamma of Eqs. (14)/(15)
    ("charging_base_multiplier", 0.0,  100.0),   # kappa of Eq. (15)
    ("kernel_n_sigma",           2.0,   5.0),   # kernel truncation, rounded to int
    ("distance_between_drone_norm", 10.0, 1000.0),
    ("charge_margin", MIN_CHARGE_MARGIN, 0.5),  
]

##### GA parameters (mode "train") #####
POPULATION_SIZE = 50
NUMBER_OF_GENERATIONS = 20
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.05
RUNS_PER_EVALUATION = 3  # the simulation is stochastic, each individual is the average of this many runs
TRAINING_DRONE_COUNTS = (3, 5, 7, 10)  # swarm sizes each individual is evaluated on, the cost is the sum over them
GA_LOGBOOK_FILE = "ga_logbook.txt"

##### Test parameters (mode "test") #####
##### Individual found by the GA tuning. Placeholder: the protocol defaults,
##### in the GENE_BOUNDS order. Replace after a real tuning run.
BEST_INDIVIDUAL = [76.31, 40.14, 7.44, 52.38, 2.84, 455.63, 0.26]
NUMBER_OF_TEST_RUNS = 10
TEST_LOG_DIR = "/logs"
##### Plots. Only makes sense on a single test run, they slow the simulation down #####
ENABLE_MAP_PLOT = False
ENABLE_SIMULATION_PLOT = False


def unpack_individual(individual):
    """
    Maps the flat GA genome onto the protocol's tunable parameters, in the
    order declared by GENE_BOUNDS.
    """
    base_variance, alpha, gamma, kappa, n_sigma, drone_norm, charge_margin = individual
    return dict(
        base_variance=base_variance,
        alpha_variance_modifier=alpha,
        energy_gamma=gamma,
        charging_base_multiplier=kappa,
        ##### the protocol takes this one as an int
        kernel_n_sigma=int(round(n_sigma)),
        distance_between_drone_norm=drone_norm,
        chage_margin=charge_margin

    )


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
    drones CHARGING above nc_charging_base, also in drone*second.
    """

    def __init__(self, simulation, number_of_drones: int, sample_interval: float,
                 nc_charging_base: int = NC_CHARGING_BASE):
        self.sim = simulation
        self.number_of_drones = number_of_drones
        self.sample_interval = sample_interval
        self.nc_charging_base = nc_charging_base
        self._protocols = None

        self.times = []
        self.uncertainty = []
        self.unvisited = []
        self.accomulated_uncertainty = 0.0
        self.map = None
        self.visited = None

        self.dead_drones = []
        self.charging_drones = []
        self.accomulated_dead_time = 0.0
        self.accomulated_charging_excess = 0.0

    def sample(self, time: float) -> None:
        if self._protocols is None:
            ##### The protocols only exist once the simulation has been         
            ##### initialized, which happens on its first step
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
        charging = sum(p.status == DroneStatus.CHARGING for p in self._protocols)
        self.accomulated_dead_time += dead * self.sample_interval
        self.accomulated_charging_excess += max(0, charging - self.nc_charging_base) * self.sample_interval

        self.dead_drones.append(dead)
        self.charging_drones.append(charging)

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


#### Objective function using simulation
#### GradySim function 
def create_and_run_simulation(individual, mode: str = "train",
                              enable_map_plot: bool = False,
                              enable_simulation_plot: bool = False,
                              sample_interval: float = GLOBAL_MAP_SAMPLE_INTERVAL,
                              observer_factory=None,
                              number_of_drones: int = None):
    """
    Runs one simulation with the given individual and returns
    (results_aggregator, global_map): the per drone results filled in by
    finish() and the GlobalMapMonitor sampled during the run. The behavior is
    the same for both modes, the only difference is that "test" writes the
    detailed logs, based on the recorder.py, and may draw the plots. While the train
    is used for the GA optimization.

    The simulation is driven event by event so the global map can be sampled
    every sample_interval seconds, see run_stepped_simulation.

    observer_factory, when given, is called with the built simulation and must
    return a callable observe(time), invoked on every sample next to the global
    map monitor. main_test.py uses it to record the run.

    number_of_drones is the size of the swarm. When None, NUMBER_OF_DRONES is
    read at call time, so main_test.py can still override the module value.
    """
    ##### Configuring global parameter
    global how_many_simulations
    how_many_simulations += 1

    if number_of_drones is None:
        number_of_drones = NUMBER_OF_DRONES

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
        max_speed_xy=10.0,
        max_speed_z=10.0,
        max_acc_xy=3.0,
        max_acc_z=3.0,
        send_telemetry=True,
    )
    builder.add_handler(DynamicVelocityMobilityHandler(mobility_config))
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=TRANSMISSION_RANGE
    )))

    if enable_simulation_plot:
        builder.add_handler(VisualizationHandler())

    results_aggregator = {}
    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=UNCERTAINTY_RATE,
        vanishing_update_time=VANISHING_UPDATE_TIME,
        number_of_drones=number_of_drones,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        **unpack_individual(individual),
        discharge_rate=DISCHARGE_RATE,
        charging_base_position=CHARGING_BASE_POSITION,
        min_charge_margin=MIN_CHARGE_MARGIN,  
        Nc_charging_base=NC_CHARGING_BASE,  
        results_aggregator=results_aggregator,
        mode=mode,
        enable_map_plot=enable_map_plot
    )

    for _ in range(number_of_drones):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

    # Building & starting
    simulation = builder.build()

    global_map = GlobalMapMonitor(simulation, number_of_drones, sample_interval, NC_CHARGING_BASE)
    observers = [global_map.sample]
    if observer_factory is not None:
        observers.append(observer_factory(simulation))

    run_stepped_simulation(simulation, sample_interval, observers)

    return results_aggregator, global_map


def evaluate_simulation_cost(results_aggregator, global_map: GlobalMapMonitor,
                             mode: str = "train"):
    """
    Turns the result of the simulation into the single value to be minimized.
    """
    ##### Getting the results of the simulation #####
    ##### The swarm size comes from the run itself, it changes during the tuning #####
    number_of_drones = global_map.number_of_drones
    medium_uncertainty = 0
    for i in range(number_of_drones):
        medium_uncertainty += results_aggregator[i]['accomulated_uncertainty']/number_of_drones

    ##### Penalties, accumulated by the monitor on every sample #####
    dead_penalty = DEAD_DRONE_PENALTY * global_map.accomulated_dead_time
    charging_penalty = CHARGING_CONGESTION_PENALTY * global_map.accomulated_charging_excess

    ##### Cost for optimization #####
    total_cost = global_map.accomulated_uncertainty*0.01 + dead_penalty + charging_penalty

    if mode == "test":
        logging.info(f"Global accomulated uncertainty: {global_map.accomulated_uncertainty}")
        logging.info(f"Global final uncertainty: {global_map.final_uncertainty}")
        logging.info(f"Global unvisited cells: {global_map.unvisited_cells}")
        logging.info(f"Medium accomulated uncertainty per drone: {medium_uncertainty}")
        logging.info(f"Dead drone time: {global_map.accomulated_dead_time} drone*s, penalty: {dead_penalty}")
        logging.info(f"Charging above Nc: {global_map.accomulated_charging_excess} drone*s, penalty: {charging_penalty}")
        logging.info(f"Cost: {total_cost}")

    return total_cost


########### GA part ##########
def objective_function(individual):
    if not is_feasible(individual):
        return 1000000.0,  # Return a large cost for infeasible solutions

    ##### The wind and the initial destinations are random, so a single run is a  #####
    ##### noisy cost: the same individual can go from 52k to 93k. The individual   #####
    ##### is evaluated by the average of RUNS_PER_EVALUATION runs instead.          #####
    ##### The parameters must work for any swarm size, not only one, so this is     #####
    ##### repeated for every TRAINING_DRONE_COUNTS and the averages are summed.     #####
    run_costs = {}
    average_costs = {}
    for number_of_drones in TRAINING_DRONE_COUNTS:
        costs = []
        for _ in range(RUNS_PER_EVALUATION):
            results_aggregator, global_map = create_and_run_simulation(
                individual, mode="train", number_of_drones=number_of_drones)
            costs.append(evaluate_simulation_cost(results_aggregator, global_map, mode="train"))
        run_costs[number_of_drones] = costs
        average_costs[number_of_drones] = float(np.mean(costs))
    total_cost = sum(average_costs.values())

    ##### Single print, the evaluations run in parallel and the lines would interleave #####
    print(f"Individual: {individual}\n"
          + "".join(f"{n} drones, cost of each run: {run_costs[n]}, average: {average_costs[n]}\n"
                    for n in TRAINING_DRONE_COUNTS)
          + f"Variable to be minimized: {total_cost}")
    print(f"Total number of simulations: {how_many_simulations}")

    return total_cost,

def is_feasible(individual):
    """Every gene has to stay inside its own GENE_BOUNDS range."""
    if len(individual) != len(GENE_BOUNDS):
        return False
    for value, (_, low, high) in zip(individual, GENE_BOUNDS):
        if not (low <= value <= high):
            return False
    return True


def run_training():
    """
    Tunes the protocol parameters with a GA. Every individual is evaluated by
    a full simulation, so the evaluations run in parallel and the protocol
    keeps its per-routine logs disabled.
    """
    ### Defining the GA ###
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) ## Minimize the accomulated uncertainty
    creator.create("Individual", list,  fitness=creator.FitnessMin) ## individual

    toolbox = base.Toolbox()

    ##### Each gene is drawn inside its own range, they are not commensurable #####
    def random_individual():
        return creator.Individual(random.uniform(low, high) for _, low, high in GENE_BOUNDS)

    toolbox.register("individual", random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    ##### Mutation step scaled per gene, a single sigma would be meaningless #####
    mutation_sigmas = [(high - low) * 0.1 for _, low, high in GENE_BOUNDS]

    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation_sigmas, indpb=0.05)

    ##### The Gaussian mutation can push a gene out of its GENE_BOUNDS range. Such an #####
    ##### individual would cost 1e6 in objective_function, which is less than a drone #####
    ##### dead for 100 s, so it would be preferred to a valid one. Clamping it back    #####
    ##### keeps every individual valid.                                                #####
    def clamp_to_bounds(mutate):
        def wrapper(*args, **kwargs):
            offspring = mutate(*args, **kwargs)
            for child in offspring:
                for k, (_, low, high) in enumerate(GENE_BOUNDS):
                    child[k] = min(max(child[k], low), high)
            return offspring
        return wrapper

    toolbox.decorate("mutate", clamp_to_bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)

    ### Parallelization
    pool = multiprocessing.Pool(processes=CORES_TO_USE)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, log = algorithms.eaSimple(pop, toolbox,
                                       cxpb=CROSSOVER_PROBABILITY,
                                       mutpb=MUTATION_PROBABILITY,
                                       ngen=NUMBER_OF_GENERATIONS,
                                       stats=stats, halloffame=hof, verbose=True)
    finally:
        pool.close()
        pool.join()

    print("=== Final Results ===")
    print(log)

    with open(GA_LOGBOOK_FILE, "w") as f:
        # Use str(log) to get the Logbook content as a string
        f.write(str(log))

    print("=== Top Best Individuals ===")
    for rank, individual in enumerate(hof):
        print(f"Rank {rank + 1}:")
        print(f"Fitness: {individual.fitness.values[0]}")
        print(f"Parameters: {individual}\n")


def run_test():
    """
    Repeats the simulation with the individual already tuned, writing the
    detailed logs so the results can be analyzed.
    """
    os.makedirs(TEST_LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(TEST_LOG_DIR, "simulation.log"),
        filemode='w',
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'
    )

    costs = []
    for run in range(NUMBER_OF_TEST_RUNS):
        logging.info(f"##### Run {run + 1} of {NUMBER_OF_TEST_RUNS} #####")

        results_aggregator, global_map = create_and_run_simulation(
            BEST_INDIVIDUAL,
            mode="test",
            enable_map_plot=ENABLE_MAP_PLOT,
            enable_simulation_plot=ENABLE_SIMULATION_PLOT
        )
        cost = evaluate_simulation_cost(results_aggregator, global_map, mode="test")
        costs.append(cost)

        print(f"Global map: {global_map.final_uncertainty:.1f} final uncertainty, "
              f"{global_map.unvisited_cells} cells never seen by the swarm")

        print(f"Run {run + 1}/{NUMBER_OF_TEST_RUNS}. Variable to be minimized: {cost}")

    print("=== Test Results ===")
    print(f"Individual: {BEST_INDIVIDUAL}")
    print(f"Runs: {NUMBER_OF_TEST_RUNS}")
    print(f"Mean cost: {np.mean(costs)}")
    print(f"Standard deviation: {np.std(costs)}")
    print(f"Min cost: {np.min(costs)}")
    print(f"Max cost: {np.max(costs)}")

    logging.info(f"Mean cost: {np.mean(costs)}, standard deviation: {np.std(costs)}")


def main():
    parser = argparse.ArgumentParser(description="Analytical coordination protocol")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train tunes the parameters with the GA, "
                             "test repeats the simulation with the tuned individual")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    else:
        run_test()


if __name__ == "__main__":
    main()
