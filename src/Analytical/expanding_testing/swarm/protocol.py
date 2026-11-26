import enum
import math
import logging
from typing import TypedDict, Type
import numpy as np
from dataclasses import dataclass
import json
import base64
from scipy.interpolate import RegularGridInterpolator
from .visualization import MapVisualizer
from .fitness import FitnessEvaluator
from .recharge_base import (BaseMessageType, BaseHeartbeatMessage, base_node_id,
                            send_landed, send_take_off)

from gradysim.protocol.interface import IProtocol
from gradysim.simulator.handler.mobility.dynamic_velocity.telemetry import DynamicVelocityTelemetry
from gradysim.protocol.plugin.battery_power import BatteryPowerPlugin, BatteryPowerConfiguration
from gradysim.protocol.messages.mobility import SetVelocityMobilityCommand
from gradysim.simulator.extension.camera import CameraHardware, CameraConfiguration
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.plugin.battery_power import BatteryPowerPlugin, BatteryPowerConfiguration


class DroneStatus(enum.Enum):
    MAPPING = 0
    GOING_TO_BASE = 1
    DEAD = 2
    CHARGING = 3
    ##### Landed on a base, waiting for the base to give it a pad (since 2026-11-24), see recharge_base.py #####
    WAITING = 4

class MessageType(enum.Enum):
    HEARTBEAT_MESSAGE = 0
    GOSSIPING_MESSAGE = 1
    SHARE_GOTO_POSITION_MESSAGE = 2
    DEAD_NOTIFICATION_MESSAGE = 3
    ##### 4 was the BASE_HEARTBEAT_MESSAGE of the bases. Since 2026-11-24 the bases are nodes         #####
    ##### (recharge_base.py), with their own message types                                            #####

class HeartBeatMessage(TypedDict):
    message_type: int
    status: int
    sender: int

class GossipingMessage(TypedDict):
    message_type: int
    map: str  # see encode_map
    sender: int
    drone_position: list
    drone_velocity: list  # sender's horizontal velocity [vx, vy] (None before its first telemetry),
                          # for the turn time of the pair decision (FitnessEvaluator.turn_time)
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


def take_off_base(drone_id: int, number_of_bases: int) -> int:
    """
    Index of the base the drone takes off from. The drones are spread over the
    bases in turn, drone i on base i % number_of_bases. main.py places them there.
    """
    return drone_id % number_of_bases



##### Wind (since 2026-11-24): the old random air speed N(0, 4) m/s along the heading, but each    #####
##### drone keeps a value for 60 s before drawing the next one (piecewise constant). Drawn every     #####
##### second it averaged out over a flight (100 s of cruise: 0.94-1.08x the mean drain), held for 60 s #####
##### it does not (0.77-1.70x over 100 s, 0.86-1.25x over 1000 s), so the drain estimate is not exact. #####
##### Same distribution as before, so the mean drain and DISCHARGE_RATE are unchanged.               #####
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
        "drone_max_acceleration": 10.0,  # horizontal acceleration limit of the mobility handler, m/s^2
        "time_to_recharge": 300.0,  # s a charge lasts on a pad, TIME_TO_RECHARGE of main.py
        ##### sigma_0^2 of Eq. (11). The kernel offsets are in CELL units, so this #####
        ##### is in cells^2: 1.0 == one cell spacing, not one metre.               #####
        'base_variance': 1.0,
        'alpha_variance_modifier': 1.0,
        'energy_gamma': 1.0,
        'charging_base_multiplier': 1.0,
        ##### Normaliser of the encounter coupling term, in metres. Smaller means #####
        ##### the two drones are pushed apart harder when they meet. Above ~500   #####
        ##### the term is swamped by the reward spread and does nothing.          #####
        'distance_between_drone_norm': 50.0,
        'kernel_n_sigma': 3, # fixed, the truncation radius of the kernel in sigmas
        ##### Uncertainty of every cell at t=0. Below 1 so the first sweep of the map, #####
        ##### the same for every individual, weighs less in the cost.                  #####
        'initial_uncertainty': 0.5,
        ##### Seed of the drone's own random streams, see initialize. None is random #####
        'seed': None,
        'a_battery_modifier': 1.0,
        'b_battery_modifier': 1.0,
        'c_battery_modifier': 1.0,
        'chage_margin': 0.25, # default charge margin, tunable
        'min_charge_margin': 0.20, # minimum charge margin, fixed
        'Nc_charging_base': 2,  # drones charging at once over all the bases before the congestion term starts
        'discharge_rate': 0.001,
        'charging_base_positions': [(0.0, 0.0)],
        ##### "train" runs silent and fast, "test" logs every routine #####
        "mode": "train",
        "enable_map_plot": False,
    }

    def initialize(self) -> None:
        self._log = logging.getLogger()

        ##### Execution mode. During the GA tuning ("train") thousands of #####
        ##### simulations run in parallel, so the per-routine logs are     #####
        ##### disabled. They are only written during a "test" run.        #####
        self.MODE = self._config.get("mode", "train")
        self.VERBOSE = self.MODE == "test"

        self.drone_position = None
        self.drone_velocity = None
        self.goto_command = np.zeros(3)

        self.TIMEOUT_TO_UPDATE_DESTINATION = 10.0
        self.MOBILITY_UPDATE_TIME = 1.0
        ##### How long a charge lasts on a pad, TIME_TO_RECHARGE of main.py, the same the bases get #####
        self.TIME_TO_RECHARGE = self._config["time_to_recharge"]

        self.UNCERTAINTY_RATE = self._config["uncertainty_rate"]
        self.VANISHING_UPDATE_TIME = self._config["vanishing_update_time"]
        self.NUMBER_OF_DRONES = self._config["number_of_drones"]
        self.MAP_WIDTH = self._config["map_width"]
        self.MAP_HEIGHT = self._config["map_height"]
        self.BASE_VARIANCE = self._config["base_variance"]
        self.ALPHA_VARIANCE_MODIFIER = self._config["alpha_variance_modifier"]
        self.ENERGY_GAMMA = self._config["energy_gamma"]
        self.CHARGING_BASE_MULTIPLIER = self._config["charging_base_multiplier"]
        self.DISTANCE_BETWEEN_DRONE_NORM = self._config["distance_between_drone_norm"]
        self.KERNEL_N_SIGMA = self._config["kernel_n_sigma"]
        self.DISCHARGE_RATE = self._config["discharge_rate"]
        self.CHARGING_BASE_POSITIONS = [tuple(position) for position in self._config["charging_base_positions"]]
        self.CHARGE_MARGIN = self._config["chage_margin"]
        self.MIN_CHARGE_MARGIN = self._config["min_charge_margin"]
        self.Nc_CHARGING_BASE = self._config["Nc_charging_base"]
        self.A_BATTERY_MODIFIER = self._config["a_battery_modifier"]
        self.B_BATTERY_MODIFIER = self._config["b_battery_modifier"]
        self.C_BATTERY_MODIFIER = self._config["c_battery_modifier"] 
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
        ##### of the cells to the base nearest to each, plus the time lost braking on the approach,        #####
        ##### speed / (2 a). Used to guess when a drone seen GOING_TO_BASE starts charging.                 #####
        cells_x = np.arange(self.MAP_WIDTH) * self.DISTANCE_BETWEEN_CELLS - (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        cells_y = np.arange(self.MAP_HEIGHT) * self.DISTANCE_BETWEEN_CELLS - (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2
        bases = np.asarray(self.CHARGING_BASE_POSITIONS, dtype=float)
        distance_cells_to_base = np.hypot(cells_x[:, np.newaxis, np.newaxis] - bases[:, 0],
                                          cells_y[np.newaxis, :, np.newaxis] - bases[:, 1]).min(axis=-1)
        self.EXPECTED_TIME_TO_BASE = (float(distance_cells_to_base.mean()) / self.speed_command
                                      + self.speed_command / (2 * self.BASE_APPROACH_DECELERATION))

        ### It's considered that the at any high the camera reach will be enough #####
        ##### Cluster plugins initialization #####
        self.fitness = FitnessEvaluator(map_width=self.MAP_WIDTH,
                                        map_height=self.MAP_HEIGHT,
                                        distance_between_cells = self.DISTANCE_BETWEEN_CELLS,
                                        camera_angle=self.CAMERA_ANGLE,
                                        base_variance=self.BASE_VARIANCE,
                                        energy_gamma=self.ENERGY_GAMMA,
                                        alpha_variance_modifier=self.ALPHA_VARIANCE_MODIFIER,
                                        charging_base_multiplier=self.CHARGING_BASE_MULTIPLIER,
                                        distance_between_drone_norm=self.DISTANCE_BETWEEN_DRONE_NORM,
                                        kernel_n_sigma=self.KERNEL_N_SIGMA,
                                        a_battery_modifier=self.A_BATTERY_MODIFIER,
                                        b_battery_modifier=self.B_BATTERY_MODIFIER,
                                        c_battery_modifier=self.C_BATTERY_MODIFIER,
                                        discharge_rate=self.DISCHARGE_RATE,
                                        charge_margin =self.CHARGE_MARGIN,
                                        Nc_charging_base=self.Nc_CHARGING_BASE,
                                        number_of_cells_x_y =self.CELLS_EVALUETED_FOR_PRIORITY,
                                        drone_max_acceleration=self._config["drone_max_acceleration"])
        
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
        ##### No telemetry has arrived yet, but the drones take off from a base, see main.py #####
        take_off_position = self.CHARGING_BASE_POSITIONS[take_off_base(self.provider.get_id(), len(self.CHARGING_BASE_POSITIONS))]
        command = GotoCoordsMobilityCommand(current_position=(*take_off_position, 0.0),
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
        # Time this drone's charge started and its battery then (the battery at landing, it does not change on the ground)
        self.charging_start_time = 0.0
        self.battery_at_charging_start = self.battery_status
        # Base this drone is going to or is on, -1 otherwise. See update_recharge_base. Its node is
        # base_node_id(NUMBER_OF_DRONES, recharge_base_index): the bases come after the drones
        self.recharge_base_index = -1


        ##### Starting the callbacks #####
        self.provider.schedule_timer("mobility",self.provider.current_time() + self.MOBILITY_UPDATE_TIME)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)
        self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)
        if self.MODE == "test":
            self.provider.schedule_timer("traveled_distance", self.provider.current_time() + 5)
        self.provider.schedule_timer("battery_check", self.provider.current_time() + self.BATTERY_CHECK_INTERVAL)

        ##### Visualizing the MAP #####
        ##### Only during a test run, the plot would slow down the GA tuning #####
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

        #self._log.info(f"Drone {self.provider.get_id()} is updating cells in range x: [{x_min}, {x_max}), y: [{y_min}, {y_max}) based on its position {self.drone_position} and observation radius {observation_radius}")

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
            int: number of the other drones still alive estimated to be on a base, charging or in the
                 waiting room (since 2026-11-24), see estimate_drone_battery
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
        # calculate the number of the other drones still alive estimated to be on a base, charging or waiting:
        # the drones a newcomer finds there, so N + 1 - Nc is the waiting it would make (over all the bases)
        drones_charging = np.sum(is_charging[alive_others])

        return mean_battery_status, drones_charging

    def estimate_drone_battery(self, battery: float, observation_time: float, status: float, current_time: float) -> tuple[float, bool]:
        """
        Estimates the battery of another drone at current_time, and whether it is on a base (charging, or
        waiting for a pad), from the last observation of it.

        The drone is assumed to fly discharging at DISCHARGE_RATE, to charge at the base for TIME_TO_RECHARGE and
        to leave it with a full battery. When the charge starts depends on the status of the observation:
        - CHARGING: the row holds the time its charge started and its battery then (BaseHeartbeatMessage)
        - WAITING: the row holds the time its charge will start, worked out from the waiting room the base
          announced (see handle_base_heartbeat), and its battery at landing. Until then it is on the ground,
          its battery does not change
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
            bool: True if the drone is estimated to be on a base at current_time, charging or waiting
        """
        while True:
            if status == DroneStatus.CHARGING.value or status == DroneStatus.WAITING.value:
                charging_start_time = observation_time
                battery_at_charging_start = battery
            elif status == DroneStatus.GOING_TO_BASE.value:
                charging_start_time = observation_time + self.EXPECTED_TIME_TO_BASE
                battery_at_charging_start = max(0.0, battery - self.DISCHARGE_RATE * self.EXPECTED_TIME_TO_BASE)
            else:
                time_to_min_charge = max(0.0, (battery - self.MIN_CHARGE_MARGIN) / self.DISCHARGE_RATE)
                charging_start_time = observation_time + time_to_min_charge
                battery_at_charging_start = min(battery, self.MIN_CHARGE_MARGIN)

            if current_time < charging_start_time:
                # In the waiting room, on the ground: the battery is the one it landed with
                if status == DroneStatus.WAITING.value:
                    return battery, True
                # Still flying, discharging since the observation
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

         
    def horizontal_velocity(self):
        """[vx, vy] of the last telemetry, or None before the first one (then the turn time is 0)."""
        if self.drone_velocity is None:
            return None
        return [float(self.drone_velocity[0]), float(self.drone_velocity[1])]

    ##### Self mobility command. When the drone reaches the destination, it calculates the next one #####
    def internal_mobility_command(self):
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2

        mean_battery, N_charging = self.aproximate_battery_consumption_model(self.swarm_battery_status, self.provider.current_time())

        target_coords, value, command = self.fitness.choose_one_cell(
            self.map[:, :, 0],
            self.drone_position, 
            self.CHARGING_BASE_POSITIONS,
            self.battery.battery_status,
            self.speed_command,
            map_center_offset=map_center_offset,
            mean_battery_value = mean_battery,
            drones_charging = N_charging,
            drone_velocity=self.horizontal_velocity()
        )

        if command == "going_to_base":
            self.status = DroneStatus.GOING_TO_BASE
        else:
            self.status = DroneStatus.MAPPING

        
        target_row, target_col = target_coords

        if self.VERBOSE:
            self._log.info(f"Drone {self.provider.get_id()} going to cell ({target_row}, {target_col})")

        #### Setting the position to go to
        x_goto = target_row * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        y_goto = target_col * self.DISTANCE_BETWEEN_CELLS - map_center_offset            
        self.goto_command = np.array([x_goto, y_goto, self.DRONE_ALTITUDE])  
        self.update_recharge_base()
        command = GotoCoordsMobilityCommand(current_position=self.drone_position,
                                            destination=self.goto_command,
                                            speed=self.speed_command)
        self.provider.send_mobility_command(command)

    
    ##### External mobility command. When receiving encountering another drone, the one with highest ID calculates the new destinations #####
    def external_mobility_command(self, another_drone_position: list, another_drone_id: int,
                                  another_drone_velocity: list = None) -> tuple[np.ndarray, DroneStatus]:
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2

        # transform list to tuple
        another_drone_position = tuple(another_drone_position)

        mean_battery, N_charging = self.aproximate_battery_consumption_model(self.swarm_battery_status, self.provider.current_time())

        destinations, best_fitness, commands = self.fitness.choose_two_cells(
            self.map[:, :, 0],
            self.drone_position,
            another_drone_position,
            self.CHARGING_BASE_POSITIONS,
            self.battery.battery_status,
            self.swarm_battery_status[another_drone_id, 0],
            self.speed_command,
            map_center_offset=map_center_offset,
            mean_battery_value = mean_battery,
            drones_charging = N_charging,
            first_drone_velocity=self.horizontal_velocity(),
            second_drone_velocity=another_drone_velocity
        )

        if commands[0] == "going_to_base":
            self.status = DroneStatus.GOING_TO_BASE
        else:
            self.status = DroneStatus.MAPPING

        ### Status of the second drone, sent to it together with its destination
        another_drone_status = DroneStatus.GOING_TO_BASE if commands[1] == "going_to_base" else DroneStatus.MAPPING

        target_row_1, target_col_1 = destinations[0]
        target_row_2, target_col_2 = destinations[1]

        if self.VERBOSE:
            self._log.info(f"Drone {self.provider.get_id()} going to cell ({target_row_1}, {target_col_1}) and sending drone to cell ({target_row_2}, {target_col_2}).")

        #### Setting position to go to
        x_goto = target_row_1 * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        y_goto = target_col_1 * self.DISTANCE_BETWEEN_CELLS - map_center_offset            
        self.goto_command = np.array([x_goto, y_goto, self.DRONE_ALTITUDE])
        self.update_recharge_base()

        ### Going to the next point            
        command = GotoCoordsMobilityCommand(current_position=self.drone_position,
                                            destination=self.goto_command,
                                            speed=self.speed_command)
        self.provider.send_mobility_command(command)

        ### Returning the values for the second drone
        x_send_command = target_row_2 * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        y_send_command = target_col_2 * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        send_command = np.array([x_send_command, y_send_command, self.DRONE_ALTITUDE])
        
        return send_command, another_drone_status  # Return also the status for the second drone

    def update_recharge_base(self):
        """
        Sets recharge_base_index after a new destination: while GOING_TO_BASE the destination is the cell of the
        base the drone chose (the one nearest to it when it chose), otherwise the drone is not going to any base.
        """
        if self.status == DroneStatus.GOING_TO_BASE:
            self.recharge_base_index = self.fitness.nearest_base(self.goto_command, self.CHARGING_BASE_POSITIONS)
        else:
            self.recharge_base_index = -1

    def send_heartbeat(self):
        #self._log.info(f"Sending heartbeat ...")
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
                'drone_velocity': self.horizontal_velocity(),
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

    def updated_map(self, data: dict):
        gossiping_msg: GossipingMessage = data
        updated_map = self.compare_maps(decode_map(gossiping_msg['map'], self.map.shape))
        return updated_map

    def update_battery_status(self, data: dict):
        gossiping_msg: GossipingMessage = data
        updated_battery_status = self.compare_battery_status(np.array(gossiping_msg['sender_battery_swarm_status']))
        return updated_battery_status
        
    def handle_base_heartbeat(self, base_msg: BaseHeartbeatMessage) -> bool:
        """
        Stores what a base tells about the drones on its ground, its pads and its waiting room. A
        base tells only about itself (since 2026-11-25): the other bases are known from the rows the
        drones met pass on. Returns True when a drone is new on a base: it was not counted there yet,
        or the stored stay is an older one that already ended. The congestion is counted over all the
        bases together, so which base a drone is on does not matter here.

        A drone on a pad gets the row (battery, time its charge started, CHARGING), as before. A drone
        of a waiting room gets (battery at landing, time its charge will start, WAITING). Its base
        sends that start already made (RechargeBase.waiting_room_starts): it hands the pads over in
        landing order the instant they free, so it knows the start exactly from the charges on its
        pads and the place in line. Its
        WAITING -> CHARGING row then has the same time, and is not new. Both rows keep a time that
        only grows from one stay on the base to the next, so the gossip (newest row wins) keeps them.
        """
        my_id = self.provider.get_id()
        is_new = False
        rows = [(drone_id, start, battery, DroneStatus.CHARGING.value)
                for drone_id, start, battery in base_msg['charging']]
        rows += [(drone_id, start, battery, DroneStatus.WAITING.value)
                 for drone_id, start, battery in base_msg['waiting']]

        for drone_id, start, battery, status in rows:
            # This drone's own stay, it already knows it
            if drone_id == my_id:
                continue
            stored_status, stored_time = self.swarm_battery_status[drone_id, 2], self.swarm_battery_status[drone_id, 1]
            is_new |= (stored_status not in (DroneStatus.CHARGING.value, DroneStatus.WAITING.value)
                       or stored_time < start)

            # The row keeps the battery and the time the charge starts, not the time it was heard,
            # so aproximate_battery_consumption_model knows exactly when the charge ends
            self.swarm_battery_status[drone_id, 0] = battery
            self.swarm_battery_status[drone_id, 1] = start
            self.swarm_battery_status[drone_id, 2] = status
        return is_new

    def start_charging(self) -> None:
        """
        The drone is on a pad: it charges for TIME_TO_RECHARGE, then "recharge_done" takes it
        off. Called when its base gives it a pad (PAD_MESSAGE): at once on a free pad, or when
        its turn in the waiting room comes.
        """
        self.status = DroneStatus.CHARGING
        self.charging_start_time = self.provider.current_time()
        self.battery_at_charging_start = self.battery.battery_status

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} is on a pad of base {self.recharge_base_index} and is recharging for {self.TIME_TO_RECHARGE}")

        self.provider.schedule_timer(
                                    "recharge_done",
                                    self.provider.current_time() + self.TIME_TO_RECHARGE
                                )

    def handle_timer(self, timer: str) -> None:
        
        if timer == "vanishing_map":
                self.vanishing_map_routine()
                
                # Keep updating the uncertainty if the drone ran out of battery or is frozen on the base,
                # charging or waiting for a pad: the camera routine is not running to account for it
                if self.status in (DroneStatus.DEAD, DroneStatus.CHARGING, DroneStatus.WAITING):
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
            ##### The drone takes off: it tells its base, which gives the pad to the first drone #####
            ##### of its waiting room and stops listing this one, and the battery plugin bills    #####
            ##### the flight again.                                                               #####
            send_take_off(self.provider, base_node_id(self.NUMBER_OF_DRONES, self.recharge_base_index))
            self.battery.charge_battery_to_full()
            self.battery.resume()
            self.status = DroneStatus.MAPPING

            if self.VERBOSE:
                self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} finished recharging")

            ##### Every routine was left without a pending timer while frozen, #####
            ##### so all of them have to be started again here.                #####
            self.internal_mobility_command()
            self.provider.schedule_timer("mobility", self.provider.current_time() + self.MOBILITY_UPDATE_TIME)
            self.provider.schedule_timer("camera", self.provider.current_time() + 1.0)
            self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)
            self.provider.schedule_timer("traveled_distance", self.provider.current_time() + 2)
            self.provider.schedule_timer("battery_check", self.provider.current_time() + self.BATTERY_CHECK_INTERVAL)
            return

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
                            self.internal_mobility_command()
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
                            ##### The drone reached the base and lands. It is frozen until it takes off #####
                            ##### again: the velocity is zeroed and no timer is rescheduled, so every    #####
                            ##### routine (camera, heartbeat, mobility, traveled distance, battery      #####
                            ##### check) stops until "recharge_done". On the ground the motors are off: #####
                            ##### the battery plugin stops, so the battery does not drain while the     #####
                            ##### drone waits or charges, and it cannot die there.                       #####
                            self.provider.send_mobility_command(SetVelocityMobilityCommand(0.0, 0.0, 0.0))
                            self.battery.shutdown()

                            ##### It waits for a pad until its base gives it one (PAD_MESSAGE): at the #####
                            ##### same instant when one is free, otherwise when its turn comes. Every   #####
                            ##### base has its pads, see recharge_base.py.                             #####
                            self.status = DroneStatus.WAITING
                            if self.VERBOSE:
                                self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} reached base {self.recharge_base_index} and asks for a pad")
                            send_landed(self.provider, base_node_id(self.NUMBER_OF_DRONES, self.recharge_base_index),
                                        self.battery.battery_status)
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
                    self.recharge_base_index = -1

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

        ##### The messages of the bases are handled here for every status, they must not reach the status gate below #####
        if msg_type == BaseMessageType.PAD_MESSAGE.value:
            # Its base gives this drone a pad: the charge starts now
            if self.status == DroneStatus.WAITING:
                self.start_charging()
            return

        if msg_type == BaseMessageType.BASE_HEARTBEAT_MESSAGE.value:
            # A base sends the drones on its own ground (see recharge_base.py), the other bases are known from
            # the drones met. It is repeated every second while a drone is on the ground.
            # When a drone is new there, the number of drones on the bases changed, so the destination is chosen
            # again with it. A drone on the ground or DEAD only stores it, it has no destination to change
            is_new_on_base = self.handle_base_heartbeat(data)
            if is_new_on_base and (self.status == DroneStatus.MAPPING or self.status == DroneStatus.GOING_TO_BASE):
                self.internal_mobility_command()
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

                        #self._log.info(f"Received map from drone {data['sender']}. My position: {self.drone_position}, other drone position: {another_drone_position}")

                        #self._log.info(f"Node {self.provider.get_id()} is calculating the new destinations")
                        send_command, another_drone_status = self.external_mobility_command(data['drone_position'], data['sender'], data.get('drone_velocity'))

                        self.send_goto_command(send_command, data['sender'], another_drone_status)
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
                #self._log.info(f"Received goto command from {goto_msg['sender']}. Going to {goto_msg['goto']}")

                self.status = DroneStatus(goto_msg['status'])

                self.goto_command = np.array(goto_msg['goto'], dtype=float)
                self.update_recharge_base()
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
    time_to_recharge: float,             # fixed, s
    base_variance: float,                # tunable, in cells^2
    alpha_variance_modifier: float,      # tunable
    energy_gamma: float,                 # tunable
    charging_base_multiplier: float,     # tunable
    distance_between_drone_norm: float,  # tunable
    kernel_n_sigma: int,                 # fixed
    a_battery_modifier: float,           # tunable
    b_battery_modifier: float,           # tunable
    c_battery_modifier: float,           # tunable
    chage_margin: float,                 # tunable
    min_charge_margin: float,            # fixed 
    Nc_charging_base: int,               # fixed, over all the bases together
    discharge_rate: float,               # fixed
    charging_base_positions: list,       # fixed, [(x, y), ...] in metres
    results_aggregator: dict,
    mode: str = "train",
    enable_map_plot: bool = False,
    initial_uncertainty: float = 0.5,
    seed: int = None
) -> Type[Drone]:
    """
    Creates a new Drone protocol class with the specified configuration.

    mode is either "train" (silent, used by the GA tuning) or "test" (writes
    the detailed routine logs). enable_map_plot turns on the live map plot,
    which should only be used on a single test run.

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
        "time_to_recharge": time_to_recharge,
        "base_variance": base_variance,
        "alpha_variance_modifier": alpha_variance_modifier,
        "energy_gamma": energy_gamma,
        "charging_base_multiplier": charging_base_multiplier,
        "distance_between_drone_norm": distance_between_drone_norm,
        "kernel_n_sigma": kernel_n_sigma,
        "a_battery_modifier": a_battery_modifier,
        "b_battery_modifier": b_battery_modifier,
        "c_battery_modifier": c_battery_modifier,
        "chage_margin": chage_margin,
        "min_charge_margin": min_charge_margin,
        'Nc_charging_base': Nc_charging_base,
        "discharge_rate": discharge_rate,
        "charging_base_positions": charging_base_positions,
        "results_aggregator": results_aggregator,
        "mode": mode,
        "enable_map_plot": enable_map_plot,
        "initial_uncertainty": initial_uncertainty,
        "seed": seed,
    }

    # Define a new class that inherits from Drone
    class ConfiguredDrone(Drone):
        # Override the _config class attribute with our new values
        _config = config
    
    return ConfiguredDrone