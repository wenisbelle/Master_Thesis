import os
import random
import logging
import argparse
import multiprocessing
from functools import partial

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
MAP_WIDTH = 100   # cells, 20 m apart: 2000 m x 2000 m centred on (0, 0)
MAP_HEIGHT = 100
NUMBER_OF_DRONES = 3  # used by the test runs, the GA tuning uses TRAINING_DRONE_COUNTS
UNCERTAINTY_RATE = 0.001
VANISHING_UPDATE_TIME = 1.0
TRANSMISSION_RANGE = 200
DRONE_SPEED = 15.0  # cruise speed, m/s. Also the speed limit of the mobility handler
DRONE_MAX_ACCELERATION = 3.0  # horizontal acceleration limit of the mobility handler, m/s^2

##### Fixed protocol parameters, not tuned 
DISCHARGE_RATE = 0.00073  # measured: 0.000725-0.000733 per second, flying at 15 m/s with the random air speed (0.00077 at 10 m/s)
CHARGING_BASE_POSITION = (0.0, 0.0)  # base of the test runs, the GA tuning draws one per run, see sample_base_position
DISTANCE_BETWEEN_CELLS = 20  # cell spacing, m. Same as the protocol's, needed to put the base on a cell
KERNEL_N_SIGMA = 3  # kernel truncation radius, in sigmas. Above ~3 the cut-off weight is negligible
INITIAL_UNCERTAINTY = 1.0  # uncertainty of every cell at t=0, the first sweep of the map weighs less in the cost
MIN_CHARGE_MARGIN = 0.20  # fraction of battery for going near the base of recharging

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

##### GA genome layout: (name, low, high), in the order the individual stores them.
##### These bounds are PLACEHOLDERS chosen to be dimensionally sane, they have not
##### been calibrated against the scenario yet.
GENE_BOUNDS = [
    ("base_variance",            0.25,  100.0),   # sigma_0^2 of Eq. (11), in cells^2
    ("alpha_variance_modifier",  0.0,   100.0),   # alpha of Eq. (11)
    ("energy_gamma",             0.10,  100.0),   # gamma of Eqs. (14)/(15)
    ("charging_base_multiplier", 0.0,  100.0),   # kappa of Eq. (15)
    ("distance_between_drone_norm", 10.0, 1000.0),
    ("a_battery_modifier", 0.0, 10.0),
    ("b_battery_modifier", 0.0, 10.0),
    ("c_battery_modifier", 0.0, 10.0),
    ("charge_margin", MIN_CHARGE_MARGIN, 0.5),  
]

##### GA parameters (mode "train") #####
POPULATION_SIZE = 50
NUMBER_OF_GENERATIONS = 20
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.30
ELITE_SIZE = 1  # best individuals copied unchanged into the next generation
RUNS_PER_EVALUATION = 5  # the simulation is stochastic, each individual is the average of this many runs, each one with its own base
TRAINING_DRONE_COUNTS = (3, 5, 7, 10)  # swarm sizes each individual is evaluated on, the cost is the sum over them
BASE_POSITION_SIGMA = 333.0  # std deviation of the base around the map centre, m. 3 sigma reaches the map border
GA_LOGBOOK_FILE = "ga_logbook.txt"

##### Test parameters (mode "test") #####
##### Individual found by the GA tuning. Placeholder: the protocol defaults,
##### in the GENE_BOUNDS order. Replace after a real tuning run.
BEST_INDIVIDUAL =  [94.15, 93.66, 30.95, 0.02, 292.09, 3.42, 8.16, 4.82, 0.49]


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
    base_variance, alpha, gamma, kappa, drone_norm, a_modifier, b_modifier, c_modifier, charge_margin = individual
    return dict(
        base_variance=base_variance,
        alpha_variance_modifier=alpha,
        energy_gamma=gamma,
        charging_base_multiplier=kappa,
        distance_between_drone_norm=drone_norm,
        a_battery_modifier=a_modifier,
        b_battery_modifier=b_modifier,
        c_battery_modifier=c_modifier,
        chage_margin=charge_margin

    )


def sample_base_position(seed: int) -> tuple:
    """
    Position of the recharge base of the training run seeded by seed: a Gaussian
    centred in the middle of the map, with BASE_POSITION_SIGMA, snapped to the
    nearest cell centre, which is the point the drones fly to when going to the
    base. Draws outside the map are drawn again (a truncated Gaussian), clipping
    them would pile the bases up on the border.

    It only depends on the seed, so every individual evaluated on that seed gets
    the same base.
    """
    rng = np.random.default_rng(seed)
    cells = np.array([MAP_WIDTH, MAP_HEIGHT])
    map_center_offset = cells * DISTANCE_BETWEEN_CELLS / 2
    while True:
        cell = np.round((rng.normal(0.0, BASE_POSITION_SIGMA, size=2) + map_center_offset)
                        / DISTANCE_BETWEEN_CELLS)
        if np.all(cell >= 0) and np.all(cell < cells):
            return tuple(float(c) for c in cell * DISTANCE_BETWEEN_CELLS - map_center_offset)


def congestion_penalty_rate(excess: int, number_of_drones: int) -> float:
    """
    Cost per simulated second of having excess drones CHARGING above the base
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
    drones CHARGING above nc_charging_base, also in drone*second.
    accomulated_charging_penalty is the integral of congestion_penalty_rate of
    that number, the congestion term of the cost. It is not linear, so it has
    to be accumulated sample by sample.
    """

    def __init__(self, simulation, number_of_drones: int, sample_interval: float):
        self.sim = simulation
        self.number_of_drones = number_of_drones
        self.sample_interval = sample_interval
        self.nc_charging_base = int(number_of_drones/3)
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
        self.accomulated_charging_penalty = 0.0

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
        excess = max(0, charging - self.nc_charging_base)
        self.accomulated_charging_excess += excess * self.sample_interval
        self.accomulated_charging_penalty += (congestion_penalty_rate(excess, self.number_of_drones)
                                              * self.sample_interval)

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
                              number_of_drones: int = None,
                              seed: int = None,
                              charging_base_position: tuple = None):
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

    seed, when given, fixes the scenario: every drone draws its initial
    destination and its wind from its own streams, seeded by (seed, drone id),
    see the protocol's initialize. The same seed gives the same scenario to
    every individual, even after their decisions diverge. The global
    generators are seeded too, for anything else that draws from them. When
    None the run is random.

    charging_base_position is the (x, y) of the recharge base, the drones also
    take off from it. When None, CHARGING_BASE_POSITION is used.
    """
    ##### Configuring global parameter
    global how_many_simulations
    how_many_simulations += 1

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if number_of_drones is None:
        number_of_drones = NUMBER_OF_DRONES

    if charging_base_position is None:
        charging_base_position = CHARGING_BASE_POSITION

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
        charging_base_position=charging_base_position,
        min_charge_margin=MIN_CHARGE_MARGIN,  
        Nc_charging_base=int(number_of_drones/3),  
        results_aggregator=results_aggregator,
        mode=mode,
        enable_map_plot=enable_map_plot,
        kernel_n_sigma=KERNEL_N_SIGMA,
        initial_uncertainty=INITIAL_UNCERTAINTY,
        seed=seed
    )

    ##### The drones take off from the recharge base #####
    for _ in range(number_of_drones):
        builder.add_node(ConfiguredDrone, (charging_base_position[0], charging_base_position[1], 0))

    # Building & starting
    simulation = builder.build()

    global_map = GlobalMapMonitor(simulation, number_of_drones, sample_interval)
    observers = [global_map.sample]
    if observer_factory is not None:
        observers.append(observer_factory(simulation))

    run_stepped_simulation(simulation, sample_interval, observers)

    return results_aggregator, global_map


def simulation_cost_terms(global_map: GlobalMapMonitor) -> dict:
    """
    The weighted terms of the cost of one simulation, keyed by COST_TERMS. Their
    sum is the value minimized, they are kept apart so the GA can log which one
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
    ##### The swarm size comes from the run itself, it changes during the tuning #####
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
        logging.info(f"Charging above Nc: {global_map.accomulated_charging_excess} drone*s, penalty: {charging_penalty}")
        logging.info(f"Cost: {total_cost}")

    return total_cost


########### GA part ##########
def objective_function(individual, seeds, base_positions):
    """
    Returns (fitness, cost_terms): the fitness tuple and the COST_TERMS that add
    up to it, aggregated the same way. evaluate_generation stores the terms on
    the individual, so the logbook can show what the best cost is made of.
    """
    if not is_feasible(individual):
        return (1000000.0,), dict.fromkeys(COST_TERMS, float("nan"))  # Return a large cost for infeasible solutions

    ##### The wind and the initial destinations are random, so a single run is a  #####
    ##### noisy cost: the same individual can go from 52k to 93k. The individual   #####
    ##### is evaluated by the average of RUNS_PER_EVALUATION runs instead.          #####
    ##### The parameters must work for any swarm size, not only one, so this is     #####
    ##### repeated for every TRAINING_DRONE_COUNTS and the averages are summed.     #####
    ##### Run r uses seeds[r] and base_positions[r], the same for every individual #####
    ##### of a generation (common random numbers), so cost differences come from   #####
    ##### the parameters, not from the wind, start positions or base each          #####
    ##### individual happened to get. The base of run r is also the same for every #####
    ##### swarm size, every size is tested in the same RUNS_PER_EVALUATION places.  #####
    run_costs = {}
    average_costs = {}
    average_terms = {}
    for number_of_drones in TRAINING_DRONE_COUNTS:
        costs = []
        terms = []
        for r in range(RUNS_PER_EVALUATION):
            results_aggregator, global_map = create_and_run_simulation(
                individual, mode="train", number_of_drones=number_of_drones,
                seed=seeds[r] + number_of_drones,
                charging_base_position=base_positions[r])
            costs.append(evaluate_simulation_cost(results_aggregator, global_map, mode="train"))
            terms.append(simulation_cost_terms(global_map))
        run_costs[number_of_drones] = costs
        average_costs[number_of_drones] = float(np.mean(costs))
        average_terms[number_of_drones] = {term: float(np.mean([t[term] for t in terms]))
                                           for term in COST_TERMS}
    total_cost = sum(average_costs.values())
    ##### Same aggregation as total_cost, term by term, so the terms add up to it #####
    cost_terms = {term: sum(average_terms[n][term] for n in TRAINING_DRONE_COUNTS)
                  for term in COST_TERMS}

    ##### Single print, the evaluations run in parallel and the lines would interleave #####
    print(f"Individual: {individual}\n"
          + "".join(f"{n} drones, cost of each run: {run_costs[n]}, average: {average_costs[n]} "
                    f"({format_cost_terms(average_terms[n])})\n"
                    for n in TRAINING_DRONE_COUNTS)
          + f"Variable to be minimized: {total_cost} ({format_cost_terms(cost_terms)})")
    print(f"Total number of simulations: {how_many_simulations}")

    return (total_cost,), cost_terms

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
    LOW  = [low  for _, low, high in GENE_BOUNDS]
    HIGH = [high for _, low, high in GENE_BOUNDS]
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=HIGH, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=HIGH, eta=20,
                     indpb=1.0 / len(GENE_BOUNDS))

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
    ##### Forked explicitly: the workers have to inherit the module constants main() may have #####
    ##### changed. A spawned worker would import them again with their default values.        #####
    pool = multiprocessing.get_context("fork").Pool(processes=CORES_TO_USE)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    ##### min_<term> columns: the COST_TERMS of the individual that scored min, #####
    ##### they add up to it. They show which term the GA is actually fighting.  #####
    min_term_fields = [f"min_{term}" for term in COST_TERMS]
    log = tools.Logbook()
    log.header = ["gen", "nevals"] + stats.fields + min_term_fields

    def evaluate_generation(gen, population):
        ##### New scenarios every generation, shared by all its individuals (see    #####
        ##### objective_function). Everyone is re-evaluated, elites included, so a  #####
        ##### lucky score from an earlier generation does not stick.                 #####
        seeds = [random.randrange(2**31) for _ in range(RUNS_PER_EVALUATION)]
        base_positions = [sample_base_position(seed) for seed in seeds]
        print(f"Generation {gen} seeds: {seeds}, base positions: {base_positions}")
        evaluations = toolbox.map(partial(toolbox.evaluate, seeds=seeds, base_positions=base_positions),
                                  population)
        for ind, (fit, cost_terms) in zip(population, evaluations):
            ind.fitness.values = fit
            ind.cost_terms = cost_terms
        hof.update(population)
        best = tools.selBest(population, 1)[0]
        log.record(gen=gen, nevals=len(population), **stats.compile(population),
                   **{field: best.cost_terms[term] for field, term in zip(min_term_fields, COST_TERMS)})
        print(log.stream)

    ##### Own loop instead of eaSimple: eaSimple has no elitism and only evaluates #####
    ##### changed individuals, so with a noisy cost it keeps old lucky scores.     #####
    try:
        evaluate_generation(0, pop)
        for gen in range(1, NUMBER_OF_GENERATIONS + 1):
            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, ELITE_SIZE)]
            offspring = algorithms.varAnd(toolbox.select(pop, len(pop) - ELITE_SIZE), toolbox,
                                          CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)
            pop = elites + offspring
            evaluate_generation(gen, pop)
    finally:
        pool.close()
        pool.join()

    print("=== Final Results ===")
    print(log)

    ##### The Hall of Fame compares costs measured on different seeds, so it keeps #####
    ##### the luckiest evaluation. The best of the last generation was measured on #####
    ##### the same seeds as its competitors, and survived several re-evaluations.  #####
    final_best = tools.selBest(pop, 1)[0]

    with open(GA_LOGBOOK_FILE, "w") as f:
        # Use str(log) to get the Logbook content as a string
        f.write(str(log))
        f.write("\n\n=== Best Individual of the Last Generation ===\n")
        f.write(f"Fitness: {final_best.fitness.values[0]}\n")
        f.write(f"Cost terms: {format_cost_terms(final_best.cost_terms)}\n")
        for (name, _, _), value in zip(GENE_BOUNDS, final_best):
            f.write(f"{name}: {value}\n")
        f.write(f"Parameters: {list(final_best)}\n")

    print("=== Best Individual of the Last Generation ===")
    print(f"Fitness: {final_best.fitness.values[0]}")
    print(f"Cost terms: {format_cost_terms(final_best.cost_terms)}")
    print(f"Parameters: {final_best}\n")

    print("=== Top Best Individuals ===")
    for rank, individual in enumerate(hof):
        print(f"Rank {rank + 1}:")
        print(f"Fitness: {individual.fitness.values[0]}")
        print(f"Cost terms: {format_cost_terms(individual.cost_terms)}")
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
    global TRANSMISSION_RANGE, GA_LOGBOOK_FILE
    parser = argparse.ArgumentParser(description="Analytical coordination protocol")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train tunes the parameters with the GA, "
                             "test repeats the simulation with the tuned individual")
    parser.add_argument("--transmission-range", type=float, default=TRANSMISSION_RANGE,
                        help="communication range of the drones, in m (default: %(default)s)")
    parser.add_argument("--logbook", default=GA_LOGBOOK_FILE,
                        help="file the GA logbook is written to in train mode (default: %(default)s)")
    args = parser.parse_args()

    ##### Module constants, read when each simulation is built. The GA workers are #####
    ##### forked after this point (see run_training), so they get these values too. #####
    TRANSMISSION_RANGE = args.transmission_range
    GA_LOGBOOK_FILE = args.logbook
    print(f"Mode: {args.mode}, transmission range: {TRANSMISSION_RANGE:g} m, logbook: {GA_LOGBOOK_FILE}")

    if args.mode == "train":
        run_training()
    else:
        run_test()


if __name__ == "__main__":
    main()
