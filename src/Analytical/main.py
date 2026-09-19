import os
import random
import logging
import argparse
import multiprocessing

from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .protocol import drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

from gradysim.simulator.handler.mobility import (
    DynamicVelocityMobilityConfiguration,
    DynamicVelocityMobilityHandler,
)

from deap import algorithms, base, creator, tools
import numpy as np

how_many_simulations = 0
CORES_TO_USE = 16

##### Scenario, shared by the tuning and the test runs #####
SIMULATION_DURATION = 2000
MAP_WIDTH = 50
MAP_HEIGHT = 50
NUMBER_OF_DRONES = 3
UNCERTAINTY_RATE = 0.001
VANISHING_UPDATE_TIME = 1.0
TRANSMISSION_RANGE = 200

##### Fixed protocol parameters, not tuned #####
DISCHARGE_RATE = 0.001
CHARGING_BASE_POSITION = (0.0, 0.0)

##### GA genome layout: (name, low, high), in the order the individual stores them.
##### These bounds are PLACEHOLDERS chosen to be dimensionally sane, they have not
##### been calibrated against the scenario yet.
GENE_BOUNDS = [
    ("base_variance",            0.25,  9.0),   # sigma_0^2 of Eq. (11), in cells^2
    ("alpha_variance_modifier",  0.0,   4.0),   # alpha of Eq. (11)
    ("energy_gamma",             1.0,  60.0),   # gamma of Eqs. (14)/(15)
    ("charging_base_multiplier", 0.0,  10.0),   # kappa of Eq. (15)
    ("kernel_n_sigma",           2.0,   4.0),   # kernel truncation, rounded to int
    ##### Encounter coupling, in metres. Measured on this scenario: at 500 the term
    ##### changes 3% of encounters, at 100 it changes 11%, at 20 it changes 85% and
    ##### triples the mean separation of the two targets. Anything above ~500 is
    ##### indistinguishable from no coupling at all, so the useful range is low.
    ("distance_between_drone_norm", 10.0, 500.0),
]

##### GA parameters (mode "train") #####
POPULATION_SIZE = 50
NUMBER_OF_GENERATIONS = 20
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.05
GA_LOGBOOK_FILE = "ga_logbook.txt"

##### Test parameters (mode "test") #####
##### Individual found by the GA tuning. Placeholder: the protocol defaults,
##### in the GENE_BOUNDS order. Replace after a real tuning run.
BEST_INDIVIDUAL = [1.0, 1.0, 20.0, 1.0, 3.0, 50.0]
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
    base_variance, alpha, gamma, kappa, n_sigma, drone_norm = individual
    return dict(
        base_variance=base_variance,
        alpha_variance_modifier=alpha,
        energy_gamma=gamma,
        charging_base_multiplier=kappa,
        ##### the protocol takes this one as an int #####
        kernel_n_sigma=int(round(n_sigma)),
        distance_between_drone_norm=drone_norm,
    )


#### Objective function using simulation execution ####
#### GradySim function #######
def create_and_run_simulation(individual, mode: str = "train",
                              enable_map_plot: bool = False,
                              enable_simulation_plot: bool = False):
    """
    Runs one simulation with the given individual and returns the results of
    every drone. The behavior is the same for both modes, the only difference
    is that "test" writes the detailed logs and may draw the plots.
    """
    ##### Configuring global parameter
    global how_many_simulations
    how_many_simulations += 1

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
        number_of_drones=NUMBER_OF_DRONES,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        **unpack_individual(individual),
        discharge_rate=DISCHARGE_RATE,
        charging_base_position=CHARGING_BASE_POSITION,
        results_aggregator=results_aggregator,
        mode=mode,
        enable_map_plot=enable_map_plot
    )

    for _ in range(NUMBER_OF_DRONES):
        builder.add_node(ConfiguredDrone, (0, 0, 0))

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

    return results_aggregator


def evaluate_simulation_cost(results_aggregator, mode: str = "train"):
    """
    Turns the results of every drone into the single value to be minimized.
    """
    ##### Getting the results of the simulation #####
    medium_uncertainty = 0
    medium_battery_final_status = 0
    for i in range(NUMBER_OF_DRONES):
        medium_uncertainty += results_aggregator[i]['accomulated_uncertainty']/NUMBER_OF_DRONES
        medium_battery_final_status += results_aggregator[i]['final_battery_status']/NUMBER_OF_DRONES
        ##### Giving a penalty if the drone ran out of battery #####
        ##### 2 is the enum status for DEAD #####
        # FUTURE ###################
        if results_aggregator[i]['drone_status'] == 2:
            print(f"Drone ran out of battery")

    medium_battery_consumption = 1.0 - medium_battery_final_status

    ##### Cost for optimization #####
    total_cost = medium_uncertainty*0.01

    if mode == "test":
        logging.info(f"Medium accomulated uncertainty: {medium_uncertainty}")
        logging.info(f"Medium battery consumption: {medium_battery_consumption}")
        logging.info(f"Cost: {total_cost}")

    return total_cost


########### GA part ##########
def objective_function(individual):
    if not is_feasible(individual):
        return 1000000.0,  # Return a large cost for infeasible solutions

    results_aggregator = create_and_run_simulation(individual, mode="train")
    total_cost = evaluate_simulation_cost(results_aggregator, mode="train")

    print(f"Individual: {individual}")
    print(f"Variable to be minimized: {total_cost}")
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

        results_aggregator = create_and_run_simulation(
            BEST_INDIVIDUAL,
            mode="test",
            enable_map_plot=ENABLE_MAP_PLOT,
            enable_simulation_plot=ENABLE_SIMULATION_PLOT
        )
        cost = evaluate_simulation_cost(results_aggregator, mode="test")
        costs.append(cost)

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
