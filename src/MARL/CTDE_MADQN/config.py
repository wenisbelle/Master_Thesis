"""
Scenario shared by the training (train.py) and the tests (main_test.py, multiple_runs.py,
check_equivalence.py).

The scenario constants are copies of Analytical/swarm/main.py, so the MARL system
runs the same scenario as the analytical one. check_equivalence.py compares them
with the analytical ones, run it after changing either side.
"""

import dataclasses

from env.mapping_environment import MappingEnvironmentConfig, RewardWeights
from env.swarm_monitor import CostWeights

##### Scenario, copied from Analytical/swarm/main.py #####
SIMULATION_DURATION = 10000  # of a test run, the training episodes are MAX_EPISODE_LENGTH long (train.py)
MAP_WIDTH = 100   # cells, 20 m apart: 2000 m x 2000 m centred on (0, 0)
MAP_HEIGHT = 100
UNCERTAINTY_RATE = 0.001
VANISHING_UPDATE_TIME = 1.0
TRANSMISSION_RANGE = 100
DRONE_SPEED = 15.0  # cruise speed, m/s. Also the speed limit of the mobility handler
DRONE_MAX_ACCELERATION = 3.0  # horizontal acceleration limit of the mobility handler, m/s^2
DISCHARGE_RATE = 0.00067  # measured 2026-11-23 after the battery plugin body-frame fix: 0.000665-0.000667 per second, cruising at 15 m/s with the random air speed, the same along x and y (0.00073 before the fix)
CHARGING_BASE_POSITION = (0.0, 0.0)  # base of the test runs, the training draws one per episode
DISTANCE_BETWEEN_CELLS = 20  # cell spacing, m. Same as the protocol's
KERNEL_N_SIGMA = 3  # of the analytical fitness function, only for its replay
INITIAL_UNCERTAINTY = 1.0  # uncertainty of every cell at t=0
MIN_CHARGE_MARGIN = 0.20  # fraction of battery for going near the base of recharging
GLOBAL_MAP_SAMPLE_INTERVAL = 1.0  # simulated seconds between two samples of the global map
TRAINING_DRONE_COUNTS = (3, 5, 7, 10)  # swarm sizes, the GA evaluates on all of them
BASE_POSITION_SIGMA = 333.0  # std deviation of the base around the map centre, m. 3 sigma reaches the map border

##### Cost of the analytical system, only reported, see env/swarm_monitor.py #####
COST = CostWeights(
    uncertainty_weight=0.00072,          # UNCERTAINTY_WEIGHT
    dead_drone_penalty=1000.0,           # DEAD_DRONE_PENALTY
    charging_congestion_penalty=30.0,    # CHARGING_CONGESTION_PENALTY
    charging_congestion_steepness=5.0,   # CHARGING_CONGESTION_STEEPNESS
)

##### Analytical fitness function, to replay the analytical protocol (decision_backend "analytical") #####
##### GENE_BOUNDS order of Analytical/swarm/main.py, and its BEST_INDIVIDUAL                         #####
GENE_NAMES = ("base_variance", "alpha_variance_modifier", "energy_gamma", "charging_base_multiplier",
              "distance_between_drone_norm", "a_battery_modifier", "b_battery_modifier",
              "c_battery_modifier", "charge_margin")
BEST_INDIVIDUAL = [94.15, 93.66, 30.95, 0.02, 292.09, 3.42, 8.16, 4.82, 0.49]

##### Reinforcement learning #####
ACTION_MAP_SIZE = 21   # the drone chooses a cell up to 10 cells (200 m) away, or the base. 21 holds every
                       # cell the fitness function can choose, see the protocol docstring
LARGE_MAP_SIZE = 50    # the whole 100x100 map is seen averaged in 2x2 blocks
REWARD = RewardWeights(
    uncertainty_scale=0.01,            # per unit of uncertainty removed
    death_penalty=20.0,                # once, when a drone dies
    charging_congestion_penalty=0.1,   # K of congestion_penalty_rate, per second
    charging_congestion_steepness=5.0, # beta of congestion_penalty_rate
)


def unpack_individual(individual) -> dict:
    """
    The fitness parameters of an individual of Analytical/swarm (main.unpack_individual
    there), plus KERNEL_N_SIGMA: the analytical_parameters of the protocol.
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
        chage_margin=charge_margin,
        kernel_n_sigma=KERNEL_N_SIGMA,
    )


def scenario_config(**overrides) -> MappingEnvironmentConfig:
    """The environment configuration of the scenario above, with overrides."""
    config = MappingEnvironmentConfig(
        algorithm_iteration_interval=1.0,
        sample_interval=GLOBAL_MAP_SAMPLE_INTERVAL,
        num_agents_choices=TRAINING_DRONE_COUNTS,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        distance_between_cells=DISTANCE_BETWEEN_CELLS,
        large_map_size=LARGE_MAP_SIZE,
        action_map_size=ACTION_MAP_SIZE,
        uncertainty_rate=UNCERTAINTY_RATE,
        vanishing_update_time=VANISHING_UPDATE_TIME,
        initial_uncertainty=INITIAL_UNCERTAINTY,
        communication_range=TRANSMISSION_RANGE,
        drone_speed=DRONE_SPEED,
        drone_max_acceleration=DRONE_MAX_ACCELERATION,
        discharge_rate=DISCHARGE_RATE,
        min_charge_margin=MIN_CHARGE_MARGIN,
        max_episode_length=SIMULATION_DURATION,
        base_position_sigma=BASE_POSITION_SIGMA,
        reward=REWARD,
        cost=COST,
    )
    return dataclasses.replace(config, **overrides)
