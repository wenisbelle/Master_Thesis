"""
Drone protocol of the MARL systems.

It is the protocol of Analytical/swarm (protocol.py), copied, with one change:
the drone does not choose its destination with the fitness function anymore.
Everything else is the same code: the map, the camera, the gossip, the battery
plugin, the charging base, the battery estimation of the other drones
(estimate_drone_battery), the charging and dead notifications, the encounter
rule (the drone with the highest id decides for both) and the statuses.

The analytical protocol chooses a destination in three places. Here each one
asks for a decision instead (request_decision), and the decision is applied
when it arrives (apply_decision):

    analytical                               here, DecisionKind
    internal_mobility_command()              OWN: the drone chooses its own destination. On
                                             reaching its destination, after a recharge and when
                                             it hears a new drone charging.
    external_mobility_command() of the       ENCOUNTER: the drone with the highest id chooses its
    drone with the highest id                own destination first, then
                                             ENCOUNTER_PARTNER: it chooses the destination of the
                                             other drone, knowing its own choice, and sends it
                                             with SHARE_GOTO_POSITION_MESSAGE, as before.

The requests go into a list shared with the environment (decision_requests,
see env/mapping_environment.py). The environment stops the simulation right
after the event that made the request, asks the policy and applies the action
before the next event, so for the simulation the decision is taken in the same
instant as in the analytical protocol.

An action is a cell of the ACTION_MAP_SIZE x ACTION_MAP_SIZE window centred
on the drone, index row * ACTION_MAP_SIZE + col, or the base, index
ACTION_MAP_SIZE**2 (see decode_action). With ACTION_MAP_SIZE = 21 the window
holds every cell the fitness function can choose ([-10, 10) around the drone),
so analytical_action can replay the analytical protocol through the same
path, which is how the equivalence with Analytical/swarm is tested.

observe() builds what the neural network sees, from the knowledge of the
drone that decides (its map and its swarm_battery_status). The old MARL
protocol kept the last known position of every drone with a timeout
(drone_states, BROADCAST_DESTINATION_MESSAGE): that is gone, the drones only
know about the others what the analytical protocol knows.
"""

import enum
import math
import logging
from typing import TypedDict, Type, Optional
import numpy as np
from dataclasses import dataclass
import json
import base64
from .visualization import MapVisualizer
from .fitness import FitnessEvaluator

from gradysim.protocol.interface import IProtocol
from gradysim.simulator.handler.mobility.dynamic_velocity.telemetry import DynamicVelocityTelemetry
from gradysim.protocol.plugin.battery_power import BatteryPowerPlugin, BatteryPowerConfiguration
from gradysim.protocol.messages.mobility import SetVelocityMobilityCommand
from gradysim.simulator.extension.camera import CameraHardware, CameraConfiguration
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand


class DroneStatus(enum.Enum):
    MAPPING = 0
    GOING_TO_BASE = 1
    DEAD = 2
    CHARGING = 3

class MessageType(enum.Enum):
    HEARTBEAT_MESSAGE = 0
    GOSSIPING_MESSAGE = 1
    SHARE_GOTO_POSITION_MESSAGE = 2
    DEAD_NOTIFICATION_MESSAGE = 3
    CHARGING_NOTIFICATION_MESSAGE = 4

class HeartBeatMessage(TypedDict):
    message_type: int
    status: int
    sender: int

class GossipingMessage(TypedDict):
    message_type: int
    map: str  # see encode_map
    sender: int
    drone_position: list
    drone_status: int
    sender_battery_swarm_status: list  # one row per drone: [battery, time of the observation, status]

class SendGoToMessage(TypedDict):
    message_type: int
    goto: list
    sender: int
    status: int  # DroneStatus value the receiver takes with the new destination

class DeadNotificationMessage(TypedDict):
    message_type: int
    sender: int
    dead_drone_id: int

class ChargingNotificationMessage(TypedDict):
    message_type: int
    sender: int
    charging_start_time: float  # time the sender reached the base, its charge ends TIME_TO_RECHARGE later
    battery: float  # battery of the sender when it reached the base


##### Decisions. See the module docstring. #####
class DecisionKind(enum.Enum):
    OWN = 0                # the drone chooses its own destination
    ENCOUNTER = 1          # the highest id of an encounter chooses its own destination...
    ENCOUNTER_PARTNER = 2  # ...and then the one of the other drone


@dataclass
class DecisionRequest:
    """
    One decision to take. decider_id is the drone whose protocol takes it, with its
    own map and battery table. subject_id is the drone that flies to the destination
    chosen: the decider itself, or the other drone of the encounter for
    ENCOUNTER_PARTNER. partner_* describe the other drone of an encounter as the decider
    knows it: for ENCOUNTER the drone met, for ENCOUNTER_PARTNER the decider, with the
    destination it just chose for itself (partner_destination).
    trigger says what made the drone decide: "arrival" (reached its destination),
    "recharge_done", "charging_notification" or "encounter", and trigger_sender the
    drone whose message it was (the charging one, or the other drone of the encounter).
    """
    kind: DecisionKind
    decider_id: int
    subject_id: int
    subject_position: np.ndarray
    subject_battery: float
    subject_status: int
    time: float
    trigger: str
    trigger_sender: Optional[int] = None
    partner_id: Optional[int] = None
    partner_position: Optional[np.ndarray] = None
    partner_battery: Optional[float] = None
    partner_destination: Optional[np.ndarray] = None
    partner_status: Optional[int] = None


##### What observe() returns, the neural network input. The sizes are fixed here, the #####
##### environment builds its specs and the actor its layers from them.               #####
MAP_CHANNELS = 3  # uncertainty, distance to the drone, energy margin. See _map_channels
AGENT_STATE_FIELDS = (
    "x", "y",                          # position of the drone, [0, 1] over the map
    "base_dx", "base_dy",              # base - position, over the map side, [-1, 1]
    "battery",                         # battery of the drone
    "energy_to_base",                  # battery spent flying straight to the base
    "safe_battery",                    # battery - energy_to_base - MIN_CHARGE_MARGIN
    "going_to_base",                   # 1 when the drone already is GOING_TO_BASE
    "map_mean", "map_std",             # mean and std deviation of the uncertainty of the map
    "swarm_mean_battery",              # estimated mean battery of the other drones alive
    "swarm_charging",                  # estimated number of the other drones charging, / N
    "charging_capacity",               # Nc / N
    "swarm_known_dead",                # other drones known to be DEAD, / N
    "swarm_size",                      # N / SWARM_SIZE_NORM
)
ENCOUNTER_STATE_FIELDS = (
    "encounter",                       # 1 for ENCOUNTER and ENCOUNTER_PARTNER, 0 for OWN
    "partner_dx", "partner_dy",        # partner position - position, over the map side
    "partner_battery",                 # battery of the partner, as the decider knows it
    "partner_energy_to_base",          # battery the partner spends flying to the base
    "partner_choice_known",            # 1 for ENCOUNTER_PARTNER: the partner already chose
    "partner_goal_dx", "partner_goal_dy",  # partner destination - position, over the map side
    "partner_going_to_base",           # 1 when the partner chose the base
)
AGENT_STATE_DIM = len(AGENT_STATE_FIELDS)
ENCOUNTER_STATE_DIM = len(ENCOUNTER_STATE_FIELDS)
SWARM_SIZE_NORM = 10.0
UNCERTAINTY_CLIP = 5.0  # the uncertainty channel is (u - mean) / std, clipped to +-UNCERTAINTY_CLIP


def encode_map(map_array: np.ndarray) -> str:
    """
    The map of a GossipingMessage, as the base64 of its raw float64 bytes. Every
    received heartbeat is answered with the whole map, and as a JSON list of floats
    a 100x100 map took ~18 ms to encode and decode, the largest cost of a run.
    This takes ~1 ms and gives back the same values bit for bit.
    """
    return base64.b64encode(np.ascontiguousarray(map_array, dtype=np.float64).tobytes()).decode("ascii")


def decode_map(encoded: str, shape: tuple) -> np.ndarray:
    """Inverse of encode_map. Every drone has a map of the same shape, so the receiver uses its own."""
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float64).reshape(shape)


def GotoCoordsMobilityCommand(current_position: np.ndarray,
                              destination: np.ndarray,
                              speed: float = 10.0) -> SetVelocityMobilityCommand:
    """
    Creates a mobility command to move the drone to the specified coordinates.

    The dynamic velocity handler only accepts velocities, so the destination is
    converted into a velocity vector pointing from the current position towards
    it, with magnitude speed. If the drone already is at the destination, a
    zero velocity is returned.
    """
    ##### Before the first telemetry the drone has no known position yet, the origin is assumed. #####
    if current_position is None:
        current_position = np.zeros(3)

    direction = np.array(destination, dtype=float) - np.array(current_position, dtype=float)
    distance = np.linalg.norm(direction)

    if distance <= 10e-4:
        return SetVelocityMobilityCommand(0.0, 0.0, 0.0)

    vx, vy, vz = (direction / distance) * speed
    return SetVelocityMobilityCommand(float(vx), float(vy), float(vz))



##### Wind (since 2026-11-24): the old random air speed N(0, 4) m/s along the heading, but each    #####
##### drone keeps a value for 60 s before drawing the next one (piecewise constant). Drawn every     #####
##### second it averaged out over a flight (100 s of cruise: 0.94-1.08x the mean drain), held for 60 s #####
##### it does not (0.77-1.70x over 100 s, 0.86-1.25x over 1000 s), so the drain estimate is not exact. #####
##### Same distribution as before, so the mean drain and DISCHARGE_RATE are unchanged. Same config as  #####
##### Analytical/*/protocol.py, so with the same seed both see the same wind.                        #####
SHARED_BATTERY_CONFIG = BatteryPowerConfiguration(
    random_air_speed=True,
    random_air_speed_u=0.0,       # mean air speed, m/s
    random_air_speed_sigma=4.0,   # std deviation, m/s
    random_air_speed_hold=60.0,   # each drawn value is kept for this many seconds
)

class Drone(IProtocol):
    ### Starting plugins ###
    camera: CameraHardware
    _log: logging.Logger
    visualizer: MapVisualizer = None
    ### Variable to track how many interactions happened ###
    Number_of_Encounters: int = 0
    battery: BatteryPowerPlugin = None

    ##### Configuration for drone inheritance #####
    _config = {
        "uncertainty_rate": 0.01,
        "vanishing_update_time": 10.0,
        "number_of_drones": 3,
        "map_width": 10,
        "map_height": 10,
        "drone_speed": 10.0,  # cruise speed, m/s
        "drone_max_acceleration": 3.0,  # horizontal acceleration limit of the mobility handler, m/s^2
        ##### Uncertainty of every cell at t=0 #####
        'initial_uncertainty': 0.5,
        ##### Seed of the drone's own random streams, see initialize. None is random #####
        'seed': None,
        'min_charge_margin': 0.20, # minimum charge margin, fixed
        'Nc_charging_base': 2,
        'discharge_rate': 0.001,
        'charging_base_position': (0.0, 0.0),
        ##### Decisions, see the module docstring #####
        'decision_requests': None,  # list shared with the environment, the requests are appended to it
        'action_map_size': 21,      # side of the window of cells the drone can choose from, odd
        'large_map_size': 50,       # side of the downsampled whole map the network sees
        ##### Tunable parameters of the analytical fitness function (main.unpack_individual of     #####
        ##### Analytical/swarm, plus kernel_n_sigma). Only for analytical_action, the replay of the  #####
        ##### analytical protocol. None: no fitness function, the decisions come from the policy.    #####
        'analytical_parameters': None,
        ##### "train" runs silent and fast, "test" logs every routine #####
        "mode": "train",
        "enable_map_plot": False,
    }

    def initialize(self) -> None:
        self._log = logging.getLogger()

        ##### Execution mode. During the training ("train") many #####
        ##### simulations run in parallel, so the per-routine logs are     #####
        ##### disabled. They are only written during a "test" run.        #####
        self.MODE = self._config.get("mode", "train")
        self.VERBOSE = self.MODE == "test"

        self.drone_position = None
        self.drone_velocity = None
        self.goto_command = np.zeros(3)

        self.TIMEOUT_TO_UPDATE_DESTINATION = 10.0
        self.MOBILITY_UPDATE_TIME = 1.0
        self.TIME_TO_RECHARGE = 100

        self.UNCERTAINTY_RATE = self._config["uncertainty_rate"]
        self.VANISHING_UPDATE_TIME = self._config["vanishing_update_time"]
        self.NUMBER_OF_DRONES = self._config["number_of_drones"]
        self.MAP_WIDTH = self._config["map_width"]
        self.MAP_HEIGHT = self._config["map_height"]
        self.DISCHARGE_RATE = self._config["discharge_rate"]
        self.CHARGING_BASE_POSITION = self._config["charging_base_position"]
        self.MIN_CHARGE_MARGIN = self._config["min_charge_margin"]
        self.Nc_CHARGING_BASE = self._config["Nc_charging_base"]
        self.results_aggregator = self._config.get("results_aggregator", {})

        self.DRONE_ALTITUDE = 50.0
        self.DISTANCE_BETWEEN_CELLS = 20
        self.CAMERA_ANGLE = np.pi/6
        self.CELLS_EVALUETED_FOR_PRIORITY = 20


        ##### Initialize map #####
        self.map = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT, 2))
        self.map[:,:,0] = self._config["initial_uncertainty"]
        self.total_uncertainty = self.map[:,:,0].sum()
        self.is_cell_visited = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT))
        self.accomulated_uncertainty = 0.0

        ##### Initial state #####
        self.status = DroneStatus.MAPPING

        #### Total distance traveled ####
        self.total_distance_traveled = 0.0
        self.last_drone_position = [0.0, 0.0, 0.0]

        ##### Camera Configuration #####
        configuration = CameraConfiguration(100, 30, 180, 0)
        self.camera = CameraHardware(self, configuration)

        ##### Drone speed #####
        self.speed_command = self._config["drone_speed"]

        ##### Approach to the base. The drone has to stop on the base, but the handler brakes at most  #####
        ##### at drone_max_acceleration: zeroing the velocity on arrival, at 15 m/s, stops it up to      #####
        ##### 37.5 m past the base. The drone flies a braking profile instead, see the mobility timer:   #####
        ##### min(speed, sqrt(2 a d), d / tau), d the distance to the base. The sqrt part brakes at a,   #####
        ##### the d / tau part settles the last metres, at up to 2a where it takes over, so a is kept   #####
        ##### at 0.4 of the handler's limit. Inside BASE_APPROACH_RADIUS the drone could pass the point  #####
        ##### where it has to start braking before the next 1 s update, so it is updated more often.    #####
        self.BASE_APPROACH_DECELERATION = 0.4 * self._config["drone_max_acceleration"]
        self.BASE_APPROACH_TIME_CONSTANT = 1.0  # tau, s. Several updates long, so it does not overshoot between two
        self.BASE_APPROACH_UPDATE_TIME = 0.1
        self.BASE_APPROACH_RADIUS = (self.speed_command**2 / (2 * self.BASE_APPROACH_DECELERATION)
                                     + self.speed_command * self.MOBILITY_UPDATE_TIME)
        ##### The charge starts once the drone is on the base and almost stopped, the drift left #####
        ##### when the velocity is zeroed is BASE_ARRIVAL_SPEED^2 / (2 * max acceleration), < 2 mm. #####
        self.BASE_ARRIVAL_TOLERANCE = 0.05
        self.BASE_ARRIVAL_SPEED = 0.1

        ##### Expected flight time to the base of a drone that decided to go there, from the mean distance #####
        ##### of the cells to the base, plus the time lost braking on the approach, speed / (2 a).          #####
        ##### Used to guess when a drone seen GOING_TO_BASE starts charging.                                #####
        cells_x = np.arange(self.MAP_WIDTH) * self.DISTANCE_BETWEEN_CELLS - (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        cells_y = np.arange(self.MAP_HEIGHT) * self.DISTANCE_BETWEEN_CELLS - (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2
        distance_cells_to_base = np.hypot(cells_x[:, np.newaxis] - self.CHARGING_BASE_POSITION[0],
                                          cells_y[np.newaxis, :] - self.CHARGING_BASE_POSITION[1])
        self.EXPECTED_TIME_TO_BASE = (float(distance_cells_to_base.mean()) / self.speed_command
                                      + self.speed_command / (2 * self.BASE_APPROACH_DECELERATION))

        ##### Decisions, see the module docstring #####
        self.decision_requests = self._config["decision_requests"]
        self.ACTION_MAP_SIZE = self._config["action_map_size"]
        self.LARGE_MAP_SIZE = self._config["large_map_size"]
        self.BASE_ACTION = self.ACTION_MAP_SIZE * self.ACTION_MAP_SIZE
        if self.ACTION_MAP_SIZE % 2 == 0:
            raise ValueError("action_map_size has to be odd, the window is centred on the drone's cell")
        if self.MAP_WIDTH % self.LARGE_MAP_SIZE or self.MAP_HEIGHT % self.LARGE_MAP_SIZE:
            raise ValueError("large_map_size has to divide the map size, the map is averaged in blocks")
        self.MAP_CENTER_OFFSET = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        ##### Cell centres, x as a column and y as a row, the same as FitnessEvaluator._cell_coords #####
        self.CELLS_X = (np.arange(self.MAP_WIDTH) * self.DISTANCE_BETWEEN_CELLS - self.MAP_CENTER_OFFSET)[:, np.newaxis]
        self.CELLS_Y = (np.arange(self.MAP_HEIGHT) * self.DISTANCE_BETWEEN_CELLS - self.MAP_CENTER_OFFSET)[np.newaxis, :]
        self.DISTANCE_CELLS_TO_BASE = np.hypot(self.CELLS_X - self.CHARGING_BASE_POSITION[0],
                                               self.CELLS_Y - self.CHARGING_BASE_POSITION[1])
        self.MAP_SIDE = self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS
        self.MAP_DIAGONAL = np.hypot(self.MAP_WIDTH, self.MAP_HEIGHT) * self.DISTANCE_BETWEEN_CELLS
        ##### Destination the ENCOUNTER_PARTNER decision of an encounter was replayed with, see analytical_action #####
        self._analytical_partner_action = None

        ### It's considered that the at any high the camera reach will be enough #####
        ##### Fitness function of the analytical protocol, only to replay it (analytical_action) #####
        analytical_parameters = self._config.get("analytical_parameters")
        self.fitness = None
        if analytical_parameters is not None:
            self.fitness = FitnessEvaluator(map_width=self.MAP_WIDTH,
                                            map_height=self.MAP_HEIGHT,
                                            distance_between_cells = self.DISTANCE_BETWEEN_CELLS,
                                            camera_angle=self.CAMERA_ANGLE,
                                            base_variance=analytical_parameters["base_variance"],
                                            energy_gamma=analytical_parameters["energy_gamma"],
                                            alpha_variance_modifier=analytical_parameters["alpha_variance_modifier"],
                                            charging_base_multiplier=analytical_parameters["charging_base_multiplier"],
                                            distance_between_drone_norm=analytical_parameters["distance_between_drone_norm"],
                                            kernel_n_sigma=analytical_parameters["kernel_n_sigma"],
                                            a_battery_modifier=analytical_parameters["a_battery_modifier"],
                                            b_battery_modifier=analytical_parameters["b_battery_modifier"],
                                            c_battery_modifier=analytical_parameters["c_battery_modifier"],
                                            discharge_rate=self.DISCHARGE_RATE,
                                            charge_margin =analytical_parameters["chage_margin"],
                                            Nc_charging_base=self.Nc_CHARGING_BASE,
                                            number_of_cells_x_y =self.CELLS_EVALUETED_FOR_PRIORITY)

        ##### Communication tracking. Avoiding communications loops #####
        self.last_drone_interaction_time = np.zeros(self.NUMBER_OF_DRONES)

        ##### Initial random position #####
        ##### Each drone has its own random streams, one for the initial destination and #####
        ##### one for the wind, derived from (seed, drone id). Drawing from them never   #####
        ##### moves the stream of another drone, so two runs with the same seed keep    #####
        ##### the same scenario for every drone, even after their decisions diverge.     #####
        destination_seed, wind_seed = np.random.SeedSequence(
            None if self._config["seed"] is None else [self._config["seed"], self.provider.get_id()]
        ).spawn(2)
        destination_rng = np.random.default_rng(destination_seed)
        self.goto_command = np.array([destination_rng.uniform(-self.DISTANCE_BETWEEN_CELLS*self.MAP_WIDTH/2, self.DISTANCE_BETWEEN_CELLS*self.MAP_WIDTH/2), destination_rng.uniform(-self.DISTANCE_BETWEEN_CELLS*self.MAP_HEIGHT/2, self.DISTANCE_BETWEEN_CELLS*self.MAP_HEIGHT/2), self.DRONE_ALTITUDE])
        ##### No telemetry has arrived yet, but the drones take off from the base, see the environment #####
        command = GotoCoordsMobilityCommand(current_position=(*self.CHARGING_BASE_POSITION, 0.0),
                                            destination=self.goto_command,
                                            speed=self.speed_command)
        self.provider.send_mobility_command(command)

        #### Energy Parameters #####
        self.battery = BatteryPowerPlugin(self, SHARED_BATTERY_CONFIG, rng=np.random.default_rng(wind_seed))
        self.BATTERY_CHECK_INTERVAL = 5.0
        ##### A DEAD drone still powers its electronics, it keeps telling the others it is DEAD with this period #####
        self.DEAD_NOTIFICATION_INTERVAL = 5.0
        ##### Charge fraction below which the drone is considered lost #####
        self.DEAD_BATTERY_THRESHOLD = 0.10
        self.battery_status = self.battery.battery_status
        # Battery of the specific drone, the time of the last observation and its status (DroneStatus value)
        # Every drone starts at t=0 with the same full battery (SHARED_BATTERY_CONFIG), so that is the initial knowledge.
        # Starting from 0 would make a drone not heard yet look dead in aproximate_battery_consumption_model
        self.swarm_battery_status = np.zeros((self.NUMBER_OF_DRONES, 3))
        self.swarm_battery_status[:, 0] = self.battery_status
        self.swarm_battery_status[:, 2] = DroneStatus.MAPPING.value
        # Time this drone reached the base and its battery at that moment, sent in the ChargingNotificationMessage
        self.charging_start_time = 0.0
        self.battery_at_charging_start = self.battery_status


        ##### Starting the callbacks #####
        self.provider.schedule_timer("mobility",self.provider.current_time() + self.MOBILITY_UPDATE_TIME)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)
        self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)
        if self.MODE == "test":
            self.provider.schedule_timer("traveled_distance", self.provider.current_time() + 5)
        self.provider.schedule_timer("battery_check", self.provider.current_time() + self.BATTERY_CHECK_INTERVAL)

        ##### Visualizing the MAP #####
        ##### Only during a test run, the plot would slow down the training #####
        if self._config.get("enable_map_plot", False) and Drone.visualizer is None:
            # The visualizer is shared by every drone of the simulation.
            Drone.visualizer = MapVisualizer(num_drones=self.NUMBER_OF_DRONES,
                                             map_width=self.MAP_WIDTH,
                                             map_height=self.MAP_HEIGHT,
                                             distance_between_cells=self.DISTANCE_BETWEEN_CELLS)



    def camera_routine(self):
        ##### New Camera Routine is needed. The previous approach was too slow for running large maps.
        ### Getting the current observation radius based on the altitude and camera angle. Assuming that the drone is always looking down.
        observation_radius = self.DRONE_ALTITUDE*np.tan(self.CAMERA_ANGLE)

        ### Converting this to number of cells to check in each direction.
        cells_to_check = int(np.ceil(observation_radius / self.DISTANCE_BETWEEN_CELLS))

        ### Getting the current cell of the drone
        if self.drone_position is None:
            return
        current_x = int((self.drone_position[0] + (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2) / self.DISTANCE_BETWEEN_CELLS)
        current_y = int((self.drone_position[1] + (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2) / self.DISTANCE_BETWEEN_CELLS)

        ### Calculating the range of cells to update based on the observation radius. The range in index, so it needs to be converted to the map coordinates.
        x_min = max(0, math.floor(current_x - cells_to_check))
        x_max = min(self.MAP_WIDTH, math.floor(current_x + cells_to_check) + 1)
        y_min = max(0, math.floor(current_y - cells_to_check))
        y_max = min(self.MAP_HEIGHT, math.floor(current_y + cells_to_check) + 1)

        ### Updating the cells in the observation range
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Calculate the center coordinates of the cell
                cell_center_x = x * self.DISTANCE_BETWEEN_CELLS - (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
                cell_center_y = y * self.DISTANCE_BETWEEN_CELLS - (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2

                # Check if the cell is within the observation radius
                distance_to_cell = np.sqrt((cell_center_x - self.drone_position[0])**2 + (cell_center_y - self.drone_position[1])**2)
                if distance_to_cell <= observation_radius:
                    self.map[x, y, 0] = 0.0
                    self.map[x, y, 1] = self.provider.current_time()
                    self.is_cell_visited[x, y] = 1

        self.total_uncertainty = self.map[:,:,0].sum()
        self.accomulated_uncertainty += self.total_uncertainty

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0], [current_x, current_y])

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a accomulated uncertainty of {self.accomulated_uncertainty}")
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")

    def aproximate_battery_consumption_model(self, swarm_battery_vector: np.ndarray, current_time: float) -> tuple[float, int]:
        """
        This function approximates the battery consumption model based on the swarm battery vector and the current time.
        Used for the prediction model when the drone is making a decision

        Args:
            swarm_battery_vector (np.ndarray): A 2D array where each row represents a drone's battery status, the last update time and its status.
            current_time (float): The current simulation time.

        Returns:
            float: mean battery status of the other drones that are still alive
            int: number of the other drones still alive estimated to be charging at the base, see estimate_drone_battery
        """
        updated_battery_vector = np.copy(swarm_battery_vector)
        is_charging = np.zeros(updated_battery_vector.shape[0], dtype=bool)

        for i in range(updated_battery_vector.shape[0]):
            updated_battery_status, is_charging[i] = self.estimate_drone_battery(updated_battery_vector[i, 0],
                                                                                 updated_battery_vector[i, 1],
                                                                                 updated_battery_vector[i, 2],
                                                                                 current_time)

            updated_battery_vector[i, 0] = updated_battery_status
            updated_battery_vector[i, 1] = current_time

        # Only the other drones that are still alive enter the swarm terms:
        others = np.arange(updated_battery_vector.shape[0]) != self.provider.get_id()
        known_dead = updated_battery_vector[:, 2] == DroneStatus.DEAD.value
        alive_others = others & ~known_dead

        if not alive_others.any():
            # Nobody to compare with: the mean is the drone's own battery (no swarm shift) and nobody is charging
            return self.battery.battery_status, 0

        # calculate the mean battery status of the other drones still alive
        mean_battery_status = np.mean(updated_battery_vector[alive_others, 0])
        # calculate the number of the other drones still alive estimated to be charging at the base
        drones_charging = np.sum(is_charging[alive_others])

        return mean_battery_status, drones_charging

    def estimate_drone_battery(self, battery: float, observation_time: float, status: float, current_time: float) -> tuple[float, bool]:
        """
        Estimates the battery of another drone at current_time, and whether it is charging at the base,
        from the last observation of it.

        The drone is assumed to fly discharging at DISCHARGE_RATE, to charge at the base for TIME_TO_RECHARGE and
        to leave it with a full battery. When the charge starts depends on the status of the observation:
        - CHARGING: the row holds the time it reached the base and its battery then (ChargingNotificationMessage)
        - GOING_TO_BASE: it reaches the base EXPECTED_TIME_TO_BASE after the observation
        - MAPPING: it reaches the base when its battery reaches MIN_CHARGE_MARGIN
        After the charge the drone is MAPPING again with a full battery, and the same cycle repeats.
        DEAD drones go through the MAPPING case, the caller leaves them out.

        Args:
            battery (float): battery of the drone at the observation.
            observation_time (float): time of the observation.
            status (float): DroneStatus value of the drone at the observation.
            current_time (float): The current simulation time.

        Returns:
            float: estimated battery of the drone at current_time
            bool: True if the drone is estimated to be charging at the base at current_time
        """
        while True:
            if status == DroneStatus.CHARGING.value:
                charging_start_time = observation_time
                battery_at_charging_start = battery
            elif status == DroneStatus.GOING_TO_BASE.value:
                charging_start_time = observation_time + self.EXPECTED_TIME_TO_BASE
                battery_at_charging_start = max(0.0, battery - self.DISCHARGE_RATE * self.EXPECTED_TIME_TO_BASE)
            else:
                time_to_min_charge = max(0.0, (battery - self.MIN_CHARGE_MARGIN) / self.DISCHARGE_RATE)
                charging_start_time = observation_time + time_to_min_charge
                battery_at_charging_start = min(battery, self.MIN_CHARGE_MARGIN)

            # Still flying, discharging since the observation
            if current_time < charging_start_time:
                return max(0.0, battery - self.DISCHARGE_RATE * (current_time - observation_time)), False

            # At the base, the battery stays the one it arrived with until the charge ends
            if current_time < charging_start_time + self.TIME_TO_RECHARGE:
                return battery_at_charging_start, True

            # The charge already ended: from then on it is MAPPING with a full battery
            battery = 1.0
            observation_time = charging_start_time + self.TIME_TO_RECHARGE
            status = DroneStatus.MAPPING.value

    ##### Map updating #####
    def vanishing_map_routine(self):
        self.map[:, :, 0] = self.map[:, :, 0] + self.UNCERTAINTY_RATE

        ##### Checking if the cell was visited #####
        ##### Importante parameter. If there are unviseted cells, there will be penalizations in the algorithm #####
        self.is_cell_visited[self.map[:, :, 1] > 0.0] = 1

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, the node {self.provider.get_id()} has {self.MAP_WIDTH*self.MAP_HEIGHT - np.sum(self.is_cell_visited)} unvisited cells")


    ##### Decisions. The analytical protocol calls the fitness function in these two methods, #####
    ##### here they ask for a decision, see request_decision and apply_decision.               #####

    ##### Self mobility command. When the drone reaches the destination, it asks for the next one #####
    def internal_mobility_command(self, trigger: str = "arrival", trigger_sender: Optional[int] = None):
        self.request_decision(DecisionRequest(
            kind=DecisionKind.OWN,
            decider_id=self.provider.get_id(),
            subject_id=self.provider.get_id(),
            subject_position=np.array(self.drone_position, dtype=float),
            subject_battery=float(self.battery.battery_status),
            subject_status=self.status.value,
            time=self.provider.current_time(),
            trigger=trigger,
            trigger_sender=trigger_sender,
        ))

    ##### External mobility command. When encountering another drone, the one with highest ID asks for the new destinations #####
    def external_mobility_command(self, another_drone_position: list, another_drone_id: int,
                                  another_drone_status: int):
        self.request_decision(DecisionRequest(
            kind=DecisionKind.ENCOUNTER,
            decider_id=self.provider.get_id(),
            subject_id=self.provider.get_id(),
            subject_position=np.array(self.drone_position, dtype=float),
            subject_battery=float(self.battery.battery_status),
            subject_status=self.status.value,
            time=self.provider.current_time(),
            trigger="encounter",
            trigger_sender=another_drone_id,
            partner_id=another_drone_id,
            partner_position=np.array(another_drone_position, dtype=float),
            ##### Same battery of the other drone the analytical choose_two_cells is given #####
            partner_battery=float(self.swarm_battery_status[another_drone_id, 0]),
            partner_status=int(another_drone_status),
        ))

    def request_decision(self, request: DecisionRequest) -> None:
        """
        Hands the decision to the environment, which applies it with apply_decision
        before the next event of the simulation.
        """
        if self.decision_requests is None:
            raise RuntimeError("The drone needs a decision but has no decision_requests list. The MARL "
                               "protocol is driven by env/mapping_environment.py, which provides it")
        self.decision_requests.append(request)

    def apply_decision(self, request: DecisionRequest, action: int) -> None:
        """
        What the analytical protocol does once the fitness function returned:
        OWN and ENCOUNTER set the status and the destination of this drone and send
        the mobility command. ENCOUNTER then asks for the decision of the other drone
        (ENCOUNTER_PARTNER), which is sent to it with SHARE_GOTO_POSITION_MESSAGE.
        """
        destination, status = self.decode_action(action, request.subject_position)

        if request.kind == DecisionKind.ENCOUNTER_PARTNER:
            if self.VERBOSE:
                self._log.info(f"Drone {self.provider.get_id()} going to cell {self.position_to_cell(self.goto_command)} "
                               f"and sending drone to cell {self.position_to_cell(destination)}.")
            self.send_goto_command(destination, request.subject_id, status)
            return

        self.status = status
        if self.VERBOSE and request.kind == DecisionKind.OWN:
            self._log.info(f"Drone {self.provider.get_id()} going to cell {self.position_to_cell(destination)}")

        #### Setting the position to go to
        self.goto_command = destination
        command = GotoCoordsMobilityCommand(current_position=self.drone_position,
                                            destination=self.goto_command,
                                            speed=self.speed_command)
        self.provider.send_mobility_command(command)

        if request.kind == DecisionKind.ENCOUNTER:
            ##### Now the destination of the other drone, knowing the one just chosen #####
            self.request_decision(DecisionRequest(
                kind=DecisionKind.ENCOUNTER_PARTNER,
                decider_id=self.provider.get_id(),
                subject_id=request.partner_id,
                subject_position=request.partner_position,
                subject_battery=request.partner_battery,
                subject_status=request.partner_status,
                time=self.provider.current_time(),
                trigger="encounter",
                trigger_sender=request.partner_id,
                partner_id=self.provider.get_id(),
                partner_position=np.array(self.drone_position, dtype=float),
                partner_battery=float(self.battery.battery_status),
                partner_destination=np.array(destination, dtype=float),
                partner_status=status.value,
            ))

    ##### Actions: the cells of the ACTION_MAP_SIZE window around the drone, and the base #####
    def position_to_cell(self, position) -> tuple[int, int]:
        """
        Nearest cell centre of a position in metres, clipped to the map. Same as
        FitnessEvaluator.position_to_cell, which the analytical protocol decides with.
        """
        row = round((position[0] + self.MAP_CENTER_OFFSET) / self.DISTANCE_BETWEEN_CELLS)
        col = round((position[1] + self.MAP_CENTER_OFFSET) / self.DISTANCE_BETWEEN_CELLS)
        return (int(np.clip(row, 0, self.MAP_WIDTH - 1)), int(np.clip(col, 0, self.MAP_HEIGHT - 1)))

    def cell_to_position(self, cell) -> np.ndarray:
        """Centre of a cell, at the flight altitude. The inverse of position_to_cell."""
        return np.array([cell[0] * self.DISTANCE_BETWEEN_CELLS - self.MAP_CENTER_OFFSET,
                         cell[1] * self.DISTANCE_BETWEEN_CELLS - self.MAP_CENTER_OFFSET,
                         self.DRONE_ALTITUDE])

    def action_window(self, position) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Rows and columns of the cells of the action window of a drone at position, and its cell."""
        current_i, current_j = self.position_to_cell(position)
        half = self.ACTION_MAP_SIZE // 2
        offsets = np.arange(self.ACTION_MAP_SIZE) - half
        return current_i + offsets, current_j + offsets, (current_i, current_j)

    def valid_actions(self, position) -> np.ndarray:
        """
        (ACTION_MAP_SIZE**2 + 1,) bool: the cells of the window inside the map, except the
        cell of the drone (the analytical candidates leave it out too), and the base,
        always valid.
        """
        rows, cols, (current_i, current_j) = self.action_window(position)
        valid = (((rows >= 0) & (rows < self.MAP_WIDTH))[:, np.newaxis]
                 & ((cols >= 0) & (cols < self.MAP_HEIGHT))[np.newaxis, :])
        half = self.ACTION_MAP_SIZE // 2
        valid[half, half] = False
        return np.append(valid.reshape(-1), True)

    def decode_action(self, action: int, position) -> tuple[np.ndarray, DroneStatus]:
        """Destination in metres and status of an action taken by a drone at position."""
        action = int(action)
        if action == self.BASE_ACTION:
            ##### Same as choose_one_cell: the base is the centre of its cell #####
            base_cell = self.position_to_cell(self.CHARGING_BASE_POSITION)
            return self.cell_to_position(base_cell), DroneStatus.GOING_TO_BASE
        if not 0 <= action < self.BASE_ACTION:
            raise ValueError(f"action {action} is outside [0, {self.BASE_ACTION}]")
        row, col = divmod(action, self.ACTION_MAP_SIZE)
        rows, cols, _ = self.action_window(position)
        cell = (int(np.clip(rows[row], 0, self.MAP_WIDTH - 1)), int(np.clip(cols[col], 0, self.MAP_HEIGHT - 1)))
        return self.cell_to_position(cell), DroneStatus.MAPPING

    def encode_action(self, cell, command: str, position) -> int:
        """Inverse of decode_action, for the cell and command the fitness function returns."""
        if command == "going_to_base":
            return self.BASE_ACTION
        current_i, current_j = self.position_to_cell(position)
        half = self.ACTION_MAP_SIZE // 2
        row, col = int(cell[0]) - current_i + half, int(cell[1]) - current_j + half
        if not (0 <= row < self.ACTION_MAP_SIZE and 0 <= col < self.ACTION_MAP_SIZE):
            raise ValueError(f"cell {cell} is outside the action window of a drone at cell "
                             f"{(current_i, current_j)}, ACTION_MAP_SIZE {self.ACTION_MAP_SIZE} is too small")
        return row * self.ACTION_MAP_SIZE + col

    def analytical_action(self, request: DecisionRequest) -> int:
        """
        The action the analytical protocol takes for this request, with the same
        fitness function and the same inputs (internal_mobility_command and
        external_mobility_command of Analytical/swarm/protocol.py). The ENCOUNTER
        request chooses both destinations at once, as choose_two_cells does, and the
        one of the other drone is returned for the ENCOUNTER_PARTNER request.
        """
        if self.fitness is None:
            raise RuntimeError("analytical_action needs the analytical_parameters of the fitness function")
        if request.kind == DecisionKind.ENCOUNTER_PARTNER:
            action, self._analytical_partner_action = self._analytical_partner_action, None
            return action

        map_center_offset = self.MAP_CENTER_OFFSET
        mean_battery, N_charging = self.aproximate_battery_consumption_model(self.swarm_battery_status, self.provider.current_time())

        if request.kind == DecisionKind.OWN:
            target_coords, value, command = self.fitness.choose_one_cell(
                self.map[:, :, 0],
                self.drone_position,
                self.CHARGING_BASE_POSITION,
                self.battery.battery_status,
                self.speed_command,
                map_center_offset=map_center_offset,
                mean_battery_value = mean_battery,
                drones_charging = N_charging
            )
            return self.encode_action(target_coords, command, self.drone_position)

        another_drone_position = tuple(request.partner_position)
        destinations, best_fitness, commands = self.fitness.choose_two_cells(
            self.map[:, :, 0],
            self.drone_position,
            another_drone_position,
            self.CHARGING_BASE_POSITION,
            self.battery.battery_status,
            self.swarm_battery_status[request.partner_id, 0],
            self.speed_command,
            map_center_offset=map_center_offset,
            mean_battery_value = mean_battery,
            drones_charging = N_charging
        )
        self._analytical_partner_action = self.encode_action(destinations[1], commands[1], another_drone_position)
        return self.encode_action(destinations[0], commands[0], self.drone_position)

    ##### Observation: what the neural network sees for a request #####
    def swarm_estimate(self, subject_id: int) -> tuple[float, int, int]:
        """
        The swarm terms of a decision for subject_id, from this drone's battery table:
        estimated mean battery of the other drones alive, how many of them are
        estimated to be charging (estimate_drone_battery), and how many are known to
        be DEAD. "Other" means other than the subject. For the subject itself it is
        aproximate_battery_consumption_model. When deciding for the other drone of an
        encounter this drone is one of the others, with its real battery and status.
        """
        rows = np.copy(self.swarm_battery_status)
        now = self.provider.current_time()
        rows[self.provider.get_id()] = (self.battery.battery_status, now, self.status.value)

        estimate = np.zeros(self.NUMBER_OF_DRONES)
        charging = np.zeros(self.NUMBER_OF_DRONES, dtype=bool)
        for j in range(self.NUMBER_OF_DRONES):
            estimate[j], charging[j] = self.estimate_drone_battery(rows[j, 0], rows[j, 1], rows[j, 2], now)

        others = np.arange(self.NUMBER_OF_DRONES) != subject_id
        known_dead = rows[:, 2] == DroneStatus.DEAD.value
        alive_others = others & ~known_dead
        if not alive_others.any():
            return float("nan"), 0, int(np.sum(others & known_dead))
        return (float(np.mean(estimate[alive_others])), int(np.sum(charging[alive_others])),
                int(np.sum(others & known_dead)))

    def energy_to_base(self, position) -> float:
        """Battery spent flying straight from position to the base, at cruise speed."""
        distance = np.hypot(position[0] - self.CHARGING_BASE_POSITION[0], position[1] - self.CHARGING_BASE_POSITION[1])
        return float(distance / self.speed_command * self.DISCHARGE_RATE)

    def _map_channels(self, cells_x: np.ndarray, cells_y: np.ndarray, uncertainty: np.ndarray,
                      position, battery: float, distance_to_base: np.ndarray) -> np.ndarray:
        """
        The MAP_CHANNELS of a set of cells, for a drone at position with battery:
            0  uncertainty, (u - mean of the map) / std of the map, clipped to +-UNCERTAINTY_CLIP
            1  distance from the drone to the cell, over the map diagonal
            2  energy margin: battery left after flying to the cell and from there to the
               base, minus MIN_CHARGE_MARGIN, clipped to [-1, 1]. Negative: the drone would
               not keep its minimum reserve. The same R as energy_effect of the fitness
               function, with MIN_CHARGE_MARGIN as the margin.
        """
        mean, std = self.map[:, :, 0].mean(), self.map[:, :, 0].std()
        distance = np.hypot(cells_x - position[0], cells_y - position[1])
        energy = battery - (distance + distance_to_base) / self.speed_command * self.DISCHARGE_RATE - self.MIN_CHARGE_MARGIN
        return np.stack([
            np.clip((uncertainty - mean) / (std + 1e-4), -UNCERTAINTY_CLIP, UNCERTAINTY_CLIP),
            np.broadcast_to(distance / self.MAP_DIAGONAL, uncertainty.shape),
            np.broadcast_to(np.clip(energy, -1.0, 1.0), uncertainty.shape),
        ]).astype(np.float32)

    def observe(self, request: DecisionRequest) -> dict:
        """
        What the neural network sees for a request, from this drone's knowledge:
            large_map_patch  (MAP_CHANNELS, LARGE_MAP_SIZE, LARGE_MAP_SIZE) the whole map,
                             averaged in blocks
            small_map_patch  (MAP_CHANNELS, ACTION_MAP_SIZE, ACTION_MAP_SIZE) the action
                             window at full resolution: [c, row, col] is the cell of action
                             row * ACTION_MAP_SIZE + col. Cells outside the map have the
                             lowest uncertainty and an energy margin of -1
            agent_state      (AGENT_STATE_DIM,) see AGENT_STATE_FIELDS
            encounter_state  (ENCOUNTER_STATE_DIM,) see ENCOUNTER_STATE_FIELDS
            valid_actions    (ACTION_MAP_SIZE**2 + 1,) see valid_actions
        """
        position = np.asarray(request.subject_position, dtype=float)
        battery = float(request.subject_battery)
        uncertainty = self.map[:, :, 0]

        ##### Whole map, averaged in blocks of (MAP_WIDTH / LARGE_MAP_SIZE)^2 cells #####
        channels = self._map_channels(self.CELLS_X, self.CELLS_Y, uncertainty, position, battery,
                                      self.DISTANCE_CELLS_TO_BASE)
        L = self.LARGE_MAP_SIZE
        large = channels.reshape(MAP_CHANNELS, L, self.MAP_WIDTH // L, L, self.MAP_HEIGHT // L).mean(axis=(2, 4))

        ##### Action window, cell by cell. Its cells are computed from their own coordinates, so #####
        ##### the ones outside the map get a distance and an energy margin too                 #####
        rows, cols, _ = self.action_window(position)
        inside_rows = (rows >= 0) & (rows < self.MAP_WIDTH)
        inside_cols = (cols >= 0) & (cols < self.MAP_HEIGHT)
        window_x = (rows * self.DISTANCE_BETWEEN_CELLS - self.MAP_CENTER_OFFSET)[:, np.newaxis].astype(float)
        window_y = (cols * self.DISTANCE_BETWEEN_CELLS - self.MAP_CENTER_OFFSET)[np.newaxis, :].astype(float)
        window_uncertainty = uncertainty[np.clip(rows, 0, self.MAP_WIDTH - 1)][:, np.clip(cols, 0, self.MAP_HEIGHT - 1)]
        small = self._map_channels(window_x, window_y, window_uncertainty, position, battery,
                                   np.hypot(window_x - self.CHARGING_BASE_POSITION[0],
                                            window_y - self.CHARGING_BASE_POSITION[1]))
        outside = ~(inside_rows[:, np.newaxis] & inside_cols[np.newaxis, :])
        small[0][outside] = -UNCERTAINTY_CLIP
        small[2][outside] = -1.0

        ##### Own state and the swarm, as estimated by the drone that decides #####
        energy_to_base = self.energy_to_base(position)
        mean_battery, charging, known_dead = self.swarm_estimate(request.subject_id)
        if not np.isfinite(mean_battery):
            mean_battery = battery  # nobody else alive, as aproximate_battery_consumption_model
        n = self.NUMBER_OF_DRONES
        agent_state = np.array([
            (position[0] + self.MAP_CENTER_OFFSET) / self.MAP_SIDE,
            (position[1] + self.MAP_CENTER_OFFSET) / self.MAP_SIDE,
            (self.CHARGING_BASE_POSITION[0] - position[0]) / self.MAP_SIDE,
            (self.CHARGING_BASE_POSITION[1] - position[1]) / self.MAP_SIDE,
            battery,
            energy_to_base,
            battery - energy_to_base - self.MIN_CHARGE_MARGIN,
            float(request.subject_status == DroneStatus.GOING_TO_BASE.value),
            min(float(uncertainty.mean()), UNCERTAINTY_CLIP),
            min(float(uncertainty.std()), UNCERTAINTY_CLIP),
            mean_battery,
            charging / n,
            self.Nc_CHARGING_BASE / n,
            known_dead / n,
            n / SWARM_SIZE_NORM,
        ], dtype=np.float32)

        ##### The other drone of an encounter, zeros for OWN #####
        encounter_state = np.zeros(ENCOUNTER_STATE_DIM, dtype=np.float32)
        if request.kind != DecisionKind.OWN:
            partner = np.asarray(request.partner_position, dtype=float)
            encounter_state[0] = 1.0
            encounter_state[1:3] = (partner[:2] - position[:2]) / self.MAP_SIDE
            encounter_state[3] = request.partner_battery
            encounter_state[4] = self.energy_to_base(partner)
            if request.partner_destination is not None:
                encounter_state[5] = 1.0
                encounter_state[6:8] = (np.asarray(request.partner_destination)[:2] - position[:2]) / self.MAP_SIDE
                encounter_state[8] = float(request.partner_status == DroneStatus.GOING_TO_BASE.value)

        return {
            "large_map_patch": large.astype(np.float32),
            "small_map_patch": small,
            "agent_state": agent_state,
            "encounter_state": encounter_state,
            "valid_actions": self.valid_actions(position),
        }

    def send_heartbeat(self):
        message: HeartBeatMessage = {
            'message_type': MessageType.HEARTBEAT_MESSAGE.value,
            'status': self.status.value,
            'sender': self.provider.get_id(),
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def send_goto_command(self, send_command: np.array, destination_id: int, status: DroneStatus):
        message: SendGoToMessage = {
            'message_type': MessageType.SHARE_GOTO_POSITION_MESSAGE.value,
            'goto': send_command.tolist(),
            'sender': self.provider.get_id(),
            'status': status.value
        }
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)


    def compare_maps(self, incoming_map: np.ndarray) -> np.ndarray:
        condition = incoming_map[:, :, 1] > self.map[:, :, 1]
        condition_3d = condition[..., np.newaxis]
        return np.where(condition_3d, incoming_map, self.map)

    def compare_battery_status(self, incoming_battery_status: np.ndarray) -> np.ndarray:
        condition = incoming_battery_status[:, 1] > self.swarm_battery_status[:, 1]
        condition_2d = condition[..., np.newaxis]
        return np.where(condition_2d, incoming_battery_status, self.swarm_battery_status)

    def received_heartbeat(self, data: dict):
        heartbeat_msg: HeartBeatMessage = data

        ### Updating the agent battery timer
        self.swarm_battery_status[self.provider.get_id(), 0] = self.battery.battery_status
        self.swarm_battery_status[self.provider.get_id(), 1] = self.provider.current_time()
        self.swarm_battery_status[self.provider.get_id(), 2] = self.status.value

        message: GossipingMessage = {
                'message_type': MessageType.GOSSIPING_MESSAGE.value,
                'map': encode_map(self.map),
                'sender': self.provider.get_id(),
                'drone_position': np.array(self.drone_position).tolist(),
                'drone_status': self.status.value,
                'sender_battery_swarm_status': self.swarm_battery_status.tolist()
                }
        destination_id = heartbeat_msg['sender']
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)

    def send_dead_notification(self):
        ##### The only thing a DEAD drone still does, every DEAD_NOTIFICATION_INTERVAL. It cannot #####
        ##### fly anymore but its electronics still have power, so it keeps telling whoever is in #####
        ##### range that it is DEAD, and they stop counting it in the swarm terms.                 #####
        message: DeadNotificationMessage = {
            'message_type': MessageType.DEAD_NOTIFICATION_MESSAGE.value,
            'sender': self.provider.get_id(),
            'dead_drone_id': self.provider.get_id(),
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def send_charging_notification(self):
        message: ChargingNotificationMessage = {
            'message_type': MessageType.CHARGING_NOTIFICATION_MESSAGE.value,
            'sender': self.provider.get_id(),
            'charging_start_time': self.charging_start_time,
            'battery': self.battery_at_charging_start,
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)


    def updated_map(self, data: dict):
        gossiping_msg: GossipingMessage = data
        updated_map = self.compare_maps(decode_map(gossiping_msg['map'], self.map.shape))
        return updated_map

    def update_battery_status(self, data: dict):
        gossiping_msg: GossipingMessage = data
        updated_battery_status = self.compare_battery_status(np.array(gossiping_msg['sender_battery_swarm_status']))
        return updated_battery_status

    def handle_timer(self, timer: str) -> None:

        if timer == "vanishing_map":
                self.vanishing_map_routine()

                # Keep updating the uncertainty if the drone ran out of battery or is frozen recharging,
                # in both cases the camera routine is not running to account for it
                if self.status == DroneStatus.DEAD or self.status == DroneStatus.CHARGING:
                    self.total_uncertainty = self.map[:,:,0].sum()
                    self.accomulated_uncertainty += self.total_uncertainty

                    if self.VERBOSE:
                        self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a accomulated uncertainty of {self.accomulated_uncertainty}")
                        self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")
                self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)

        ##### Handled outside of the status gate below, the drone is DEAD when it fires #####
        if timer == "dead_notification":
            self.send_dead_notification()
            self.provider.schedule_timer("dead_notification", self.provider.current_time() + self.DEAD_NOTIFICATION_INTERVAL)
            return

        ##### Handled outside of the status gate below, the drone is CHARGING when it fires #####
        if timer == "recharge_done":
            self.battery.charge_battery_to_full()
            self.status = DroneStatus.MAPPING

            if self.VERBOSE:
                self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} finished recharging")

            ##### Every routine was left without a pending timer while frozen, #####
            ##### so all of them have to be started again here.                #####
            self.internal_mobility_command(trigger="recharge_done")
            self.provider.schedule_timer("mobility", self.provider.current_time() + self.MOBILITY_UPDATE_TIME)
            self.provider.schedule_timer("camera", self.provider.current_time() + 1.0)
            self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)
            self.provider.schedule_timer("traveled_distance", self.provider.current_time() + 2)
            self.provider.schedule_timer("battery_check", self.provider.current_time() + self.BATTERY_CHECK_INTERVAL)
            return

        if timer == "charging_notification":
            if self.status == DroneStatus.CHARGING:
                self.send_charging_notification()
                self.provider.schedule_timer("charging_notification", self.provider.current_time() + 1)

        if self.status == DroneStatus.MAPPING or self.status == DroneStatus.GOING_TO_BASE:
            if timer == "camera":
                self.camera_routine()
                self.provider.schedule_timer("camera", self.provider.current_time() + 1.0)

            if timer == "mobility":
                if self.drone_position is not None:
                    current_pos_array = np.array(self.drone_position)
                    distance_to_goto = np.linalg.norm(current_pos_array - self.goto_command)
                    next_mobility_update = self.MOBILITY_UPDATE_TIME

                    if self.status == DroneStatus.MAPPING:
                        # Keep the monitoring activity

                        ##### The tolerance has to cover the distance flown between two #####
                        ##### ticks, otherwise the drone passes the destination without #####
                        ##### ever reporting the arrival.                               #####
                        if distance_to_goto < self.speed_command * self.MOBILITY_UPDATE_TIME:
                            self.internal_mobility_command(trigger="arrival")
                        else:
                            ##### The handler only holds a velocity, so the heading has #####
                            ##### to be refreshed every tick or the drone keeps flying  #####
                            ##### in the direction of an already outdated destination.  #####
                            command = GotoCoordsMobilityCommand(current_position=current_pos_array,
                                                                destination=self.goto_command,
                                                                speed=self.speed_command)
                            self.provider.send_mobility_command(command)


                    elif self.status == DroneStatus.GOING_TO_BASE:
                        speed = 0.0 if self.drone_velocity is None else float(np.linalg.norm(self.drone_velocity))
                        if distance_to_goto < self.BASE_ARRIVAL_TOLERANCE and speed < self.BASE_ARRIVAL_SPEED:
                            ##### The drone reached the base. It is frozen for TIME_TO_RECHARGE  #####
                            ##### to simulate the recharge: the velocity is zeroed and no timer  #####
                            ##### is rescheduled, so every routine (camera, heartbeat, mobility, #####
                            ##### traveled distance, battery check) stops until "recharge_done". #####
                            self.status = DroneStatus.CHARGING
                            self.provider.send_mobility_command(SetVelocityMobilityCommand(0.0, 0.0, 0.0))
                            self.charging_start_time = self.provider.current_time()
                            self.battery_at_charging_start = self.battery.battery_status

                            if self.VERBOSE:
                                self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} reached the base and is recharging for {self.TIME_TO_RECHARGE}")

                            # Warn other drones that this one is charging, so they can update their swarm battery
                            self.provider.schedule_timer("charging_notification", self.provider.current_time() + 1)

                            self.provider.schedule_timer(
                                                        "recharge_done",
                                                        self.provider.current_time() + self.TIME_TO_RECHARGE
                                                    )
                            return
                        else:
                            ##### Same refresh as in the MAPPING branch, and for the same  #####
                            ##### reason. Without it the velocity set when the base was    #####
                            ##### chosen is never re-aimed: the drone curves under the     #####
                            ##### handler's acceleration limit, misses the arrival window, #####
                            ##### and then flies in a straight line off the map forever.   #####
                            ##### The speed follows the braking profile, so the drone stops #####
                            ##### on the base, see BASE_APPROACH_DECELERATION.              #####
                            approach_speed = min(self.speed_command,
                                                 np.sqrt(2 * self.BASE_APPROACH_DECELERATION * distance_to_goto),
                                                 distance_to_goto / self.BASE_APPROACH_TIME_CONSTANT)
                            command = GotoCoordsMobilityCommand(current_position=current_pos_array,
                                                                destination=self.goto_command,
                                                                speed=approach_speed)
                            self.provider.send_mobility_command(command)
                            if distance_to_goto < self.BASE_APPROACH_RADIUS:
                                next_mobility_update = self.BASE_APPROACH_UPDATE_TIME

                    self.provider.schedule_timer(
                                                "mobility",
                                                self.provider.current_time() + next_mobility_update
                                            )

            if timer == "heartbeat":
                self.send_heartbeat()
                self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)

            if timer == "traveled_distance":
                if self.drone_position is not None:
                    current_pos_array = np.array(self.drone_position)
                    distance_increment = np.linalg.norm(current_pos_array - self.last_drone_position)
                    self.total_distance_traveled += distance_increment
                    self.last_drone_position = current_pos_array

                    if self.VERBOSE:
                        self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} has traveled a total distance of {self.total_distance_traveled}")

                self.provider.schedule_timer("traveled_distance", self.provider.current_time() + 2)

            if timer == "battery_check":
                ##### battery_status is a property of the plugin, kept here so #####
                ##### finish() can report the charge left at the end of the run #####
                self.battery_status = self.battery.battery_status

                if self.VERBOSE:
                    self._log.info(f"At time {self.provider.current_time()} the battery status is: {self.battery_status}")

                if self.battery_status <= self.DEAD_BATTERY_THRESHOLD:
                    self._log.warning(f"Drone {self.provider.get_id()} has no battery. Drone is dead.")
                    self.status = DroneStatus.DEAD

                    ##### The drone stops where it is. The mobility handler holds the last  #####
                    ##### commanded velocity forever and no routine is rescheduled once     #####
                    ##### DEAD, so the velocity has to be zeroed explicitly or the drone    #####
                    ##### would keep drifting for the rest of the simulation.               #####
                    self.provider.send_mobility_command(SetVelocityMobilityCommand(0.0, 0.0, 0.0))
                    self.goto_command = np.array(self.drone_position)

                    ##### Stopping the drone is not enough: the battery plugin drives its    #####
                    ##### own timer, outside the status gate below, and at zero velocity the #####
                    ##### model bills hover power - the most expensive point of the curve.   #####
                    ##### Left running it reaches 0 J and raises, aborting the simulation.   #####
                    self.battery.shutdown()

                    ##### No timer is rescheduled from here on, so every routine stops. #####
                    ##### The only one started is the DEAD notification, from now on    #####
                    ##### the drone only tells the others in range that it is DEAD.     #####
                    self.send_dead_notification()
                    self.provider.schedule_timer("dead_notification", self.provider.current_time() + self.DEAD_NOTIFICATION_INTERVAL)
                    return

                self.provider.schedule_timer("battery_check", self.provider.current_time() + self.BATTERY_CHECK_INTERVAL)



    def handle_packet(self, message: str) -> None:
        data: dict = json.loads(message)

        if 'message_type' not in data:
            self._log.warning(f"Received message without a message_type: {data}")
            return

        msg_type = data['message_type']

        if msg_type == MessageType.CHARGING_NOTIFICATION_MESSAGE.value:
            charging_msg: ChargingNotificationMessage = data
            # The notification is repeated every second while the sender charges, it is only new information when
            # the sender was not counted as charging yet, or when the stored charge is an older one that already ended
            is_new_charging = (self.swarm_battery_status[charging_msg['sender'], 2] != DroneStatus.CHARGING.value
                               or self.swarm_battery_status[charging_msg['sender'], 1] < charging_msg['charging_start_time'])

            # The row keeps the battery and the time the charge started, not the time it was heard,
            # so aproximate_battery_consumption_model knows exactly when the charge ends
            self.swarm_battery_status[charging_msg['sender'], 0] = charging_msg['battery']
            self.swarm_battery_status[charging_msg['sender'], 1] = charging_msg['charging_start_time']
            self.swarm_battery_status[charging_msg['sender'], 2] = DroneStatus.CHARGING.value

            # The number of drones charging changed, so the destination is chosen again with it.
            # A CHARGING or DEAD drone only stores it, it has no destination to change
            if is_new_charging and (self.status == DroneStatus.MAPPING or self.status == DroneStatus.GOING_TO_BASE):
                self.internal_mobility_command(trigger="charging_notification", trigger_sender=charging_msg['sender'])

            # Handled here for every status, it must not reach the status gate below
            return

        if self.status == DroneStatus.MAPPING or self.status == DroneStatus.GOING_TO_BASE:

            if msg_type == MessageType.HEARTBEAT_MESSAGE.value:
                self.received_heartbeat(data)

            elif msg_type == MessageType.GOSSIPING_MESSAGE.value:
                self.map = self.updated_map(data)

                self.swarm_battery_status = self.update_battery_status(data)

                # If the drone is going to the recharge base, it can uptada its destination based on
                # the information received from the other drone
                if self.provider.current_time() - self.last_drone_interaction_time[data['sender']]  > self.TIMEOUT_TO_UPDATE_DESTINATION: # the drone id starts at 0
                    if self.provider.get_id() >= data['sender']:
                        ### Update the number of interactions ###
                        Drone.Number_of_Encounters += 1

                        ##### The destinations of both drones are asked for, see apply_decision, #####
                        ##### which sends the other one its destination                          #####
                        self.external_mobility_command(data['drone_position'], data['sender'], data['drone_status'])
                        self.last_drone_interaction_time[data['sender']] = self.provider.current_time() # the drone id starts at 0


            elif msg_type == MessageType.DEAD_NOTIFICATION_MESSAGE.value:
                # Only the swarm battery vector changes, the destination stays the same.
                # The row gets the time of this observation, so it is newer than anything the gossip still carries
                # about that drone and spreads from here with the normal gossip. The battery is kept, it is not used for a DEAD drone
                dead_msg: DeadNotificationMessage = data
                self.swarm_battery_status[dead_msg['dead_drone_id'], 1] = self.provider.current_time()
                self.swarm_battery_status[dead_msg['dead_drone_id'], 2] = DroneStatus.DEAD.value


            elif msg_type == MessageType.SHARE_GOTO_POSITION_MESSAGE.value:
                # Destination and status chosen for this drone by the one with the highest ID in the encounter.
                # Taken while MAPPING or GOING_TO_BASE, so a drone going to the base can be sent back to mapping
                goto_msg: SendGoToMessage = data

                self.status = DroneStatus(goto_msg['status'])

                self.goto_command = np.array(goto_msg['goto'], dtype=float)
                command = GotoCoordsMobilityCommand(current_position=self.drone_position,
                                                    destination=self.goto_command,
                                                    speed=self.speed_command)
                self.provider.send_mobility_command(command)

            else:
                self._log.warning(f"Received message with unknown type: {msg_type}")

    def handle_telemetry(self, telemetry: DynamicVelocityTelemetry) -> None:
        self.drone_position = telemetry.current_position
        self.drone_velocity = telemetry.current_velocity


    def finish(self) -> None:
        final_uncertainty = self.map[:,:,0].sum()
        total_cells = self.MAP_WIDTH * self.MAP_HEIGHT
        visited_cells = np.sum(self.is_cell_visited)
        unvisited_cells = total_cells - visited_cells
        self.battery_status = self.battery.battery_status

        if self.VERBOSE:
            self._log.info(f"Drone {self.provider.get_id()} final uncertainty: {final_uncertainty}, unvisited cells: {unvisited_cells}")
            self._log.info(f"Drone {self.provider.get_id()} battery status: {self.battery_status}")
            self._log.info(f"Drone {self.provider.get_id()} number of encounters: {Drone.Number_of_Encounters}")
            self._log.info(f"Drone {self.provider.get_id()} final status: {self.status.value}")

        if self.results_aggregator is not None:
            self.results_aggregator[self.provider.get_id()] = {
                "final_uncertainty": float(final_uncertainty),
                "unvisited_cells": float(unvisited_cells),
                "accomulated_uncertainty": float(self.accomulated_uncertainty),
                "total_distance_traveled": float(self.total_distance_traveled),
                "final_battery_status": float(self.battery_status),
                "drone_status": int(self.status.value)
            }


def drone_protocol_factory(
    uncertainty_rate: float,
    vanishing_update_time: float,
    number_of_drones: int,
    map_width: int,
    map_height: int,
    drone_speed: float,                  # fixed, m/s
    drone_max_acceleration: float,       # fixed, m/s^2
    min_charge_margin: float,            # fixed
    Nc_charging_base: int,               # fixed
    discharge_rate: float,               # fixed
    charging_base_position: tuple,       # fixed
    results_aggregator: dict,
    decision_requests: list,
    action_map_size: int = 21,
    large_map_size: int = 50,
    analytical_parameters: dict = None,
    mode: str = "train",
    enable_map_plot: bool = False,
    initial_uncertainty: float = 0.5,
    seed: int = None
) -> Type[Drone]:
    """
    Creates a new Drone protocol class with the specified configuration.

    mode is either "train" (silent) or "test" (writes the detailed routine logs).
    enable_map_plot turns on the live map plot, which should only be used on a
    single test run.

    decision_requests is the list the environment reads the requests from, see the
    module docstring. analytical_parameters are the tunable parameters of the
    analytical fitness function, only needed to replay the analytical protocol.

    initial_uncertainty is the uncertainty of every cell at t=0. seed seeds the
    random streams of every drone (initial destination and wind), None is random.
    """
    # Create a new configuration dictionary
    config = {
        "uncertainty_rate": uncertainty_rate,
        "vanishing_update_time": vanishing_update_time,
        "number_of_drones": number_of_drones,
        "map_width": map_width,
        "map_height": map_height,
        "drone_speed": drone_speed,
        "drone_max_acceleration": drone_max_acceleration,
        "min_charge_margin": min_charge_margin,
        'Nc_charging_base': Nc_charging_base,
        "discharge_rate": discharge_rate,
        "charging_base_position": charging_base_position,
        "results_aggregator": results_aggregator,
        "decision_requests": decision_requests,
        "action_map_size": action_map_size,
        "large_map_size": large_map_size,
        "analytical_parameters": analytical_parameters,
        "mode": mode,
        "enable_map_plot": enable_map_plot,
        "initial_uncertainty": initial_uncertainty,
        "seed": seed
    }

    # Define a new class that inherits from Drone
    class ConfiguredDrone(Drone):
        # Override the _config class attribute with our new values
        _config = config

    return ConfiguredDrone
