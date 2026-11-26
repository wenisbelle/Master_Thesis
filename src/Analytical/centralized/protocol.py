"""
The drone of the centralized system: the swarm drone of Analytical/swarm/protocol.py
with perfect, central information and a central charging scheduler.

What changes from swarm, and nothing else (see README.md):
- No radio between the drones. No heartbeat, gossip, goto, dead or charging message is
  ever sent. The only messages are the ones of a landed drone with the base node (since
  2026-11-24, recharge_base.py): it asks for a pad, gets it, and frees it at take-off.
- One global map (world.py), shared by every drone, instead of a map per drone
  merged in encounters.
- The true battery, position and status of every drone, read from the world, instead
  of the swarm battery estimate. swarm_battery_status are kept with the swarm names,
  filled with the truth, so the recorder and the plots read them as they read swarm.
- The base decision is the charging scheduler's (world.CentralizedWorld.time_to_leave),
  asked on every mobility tick. The base attraction (kappa) and the genes a, b, c and
  charge_margin are gone, the reserve is the fixed MIN_CHARGE_MARGIN.
- Encounters become pair decisions by distance: when two MAPPING drones are within
  encounter_radius, the one with the higher id chooses the destinations of both, as in
  swarm. Only MAPPING drones take part, a drone sent to the base by the scheduler keeps going.
"""

import enum
import json
import math
import logging
from typing import Type
import numpy as np

from .visualization import MapVisualizer
from .fitness import FitnessEvaluator
from .recharge_base import BaseMessageType, base_node_id, send_landed, send_take_off

from gradysim.protocol.interface import IProtocol
from gradysim.simulator.handler.mobility.dynamic_velocity.telemetry import DynamicVelocityTelemetry
from gradysim.protocol.plugin.battery_power import BatteryPowerPlugin, BatteryPowerConfiguration
from gradysim.protocol.messages.mobility import SetVelocityMobilityCommand
from gradysim.simulator.extension.camera import CameraHardware, CameraConfiguration


class DroneStatus(enum.Enum):
    MAPPING = 0
    GOING_TO_BASE = 1
    DEAD = 2
    CHARGING = 3
    ##### Landed on the base, waiting for the base to give it a pad (since 2026-11-24), see recharge_base.py #####
    WAITING = 4


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
##### Same distribution as before, so the mean drain and DISCHARGE_RATE are unchanged.               #####
SHARED_BATTERY_CONFIG = BatteryPowerConfiguration(
    battery_capacity=10000.0,     # mAh, double the 5000 default (2026-11-26). total_uav_mass unchanged, so the drain per second halves, see DISCHARGE_RATE
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
    ### Number of pair decisions, the encounters of swarm ###
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
        'base_variance': 4.0,
        'alpha_variance_modifier': 1.0,
        'energy_gamma': 1.0,
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
        'min_charge_margin': 0.25, # the reserve of the scheduler, fixed (main.py MIN_CHARGE_MARGIN)
        'Nc_charging_base': 2,
        'discharge_rate': 0.001,
        'charging_base_position': (0.0, 0.0),
        ##### Two MAPPING drones closer than this, in m, make a pair decision, see encounter_routine #####
        'encounter_radius': 100.0,
        ##### The CentralizedWorld of the simulation, shared by every drone #####
        'world': None,
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
        ##### Period of the pair decision check, the heartbeat period of swarm #####
        self.ENCOUNTER_CHECK_TIME = 1.0
        ##### How long a charge lasts on a pad, TIME_TO_RECHARGE of main.py, the same the base and the world use #####
        self.TIME_TO_RECHARGE = self._config["time_to_recharge"]

        self.UNCERTAINTY_RATE = self._config["uncertainty_rate"]
        self.VANISHING_UPDATE_TIME = self._config["vanishing_update_time"]
        self.NUMBER_OF_DRONES = self._config["number_of_drones"]
        self.MAP_WIDTH = self._config["map_width"]
        self.MAP_HEIGHT = self._config["map_height"]
        self.BASE_VARIANCE = self._config["base_variance"]
        self.ALPHA_VARIANCE_MODIFIER = self._config["alpha_variance_modifier"]
        self.ENERGY_GAMMA = self._config["energy_gamma"]
        self.DISTANCE_BETWEEN_DRONE_NORM = self._config["distance_between_drone_norm"]
        self.KERNEL_N_SIGMA = self._config["kernel_n_sigma"]
        self.DISCHARGE_RATE = self._config["discharge_rate"]
        self.CHARGING_BASE_POSITION = self._config["charging_base_position"]
        ##### The base is a node, added after the drones, see recharge_base.py. The only radio of #####
        ##### centralized: the drones read everything else from the world                         #####
        self.BASE_NODE_ID = base_node_id(self.NUMBER_OF_DRONES)
        self.MIN_CHARGE_MARGIN = self._config["min_charge_margin"]
        self.Nc_CHARGING_BASE = self._config["Nc_charging_base"]
        self.ENCOUNTER_RADIUS = self._config["encounter_radius"]
        self.results_aggregator = self._config.get("results_aggregator", {})

        self.DRONE_ALTITUDE = 50.0
        self.DISTANCE_BETWEEN_CELLS = 20
        self.CAMERA_ANGLE = np.pi/6
        self.CELLS_EVALUETED_FOR_PRIORITY = 20

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

        ##### Expected flight time to the base from the mean distance of the cells, plus the braking, #####
        ##### as in swarm. Centralized does not decide with it (the scheduler uses the real distance), #####
        ##### it is only kept for the recorder, which stores it with the run.                          #####
        cells_x = np.arange(self.MAP_WIDTH) * self.DISTANCE_BETWEEN_CELLS - (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        cells_y = np.arange(self.MAP_HEIGHT) * self.DISTANCE_BETWEEN_CELLS - (self.MAP_HEIGHT * self.DISTANCE_BETWEEN_CELLS) / 2
        distance_cells_to_base = np.hypot(cells_x[:, np.newaxis] - self.CHARGING_BASE_POSITION[0],
                                          cells_y[np.newaxis, :] - self.CHARGING_BASE_POSITION[1])
        self.EXPECTED_TIME_TO_BASE = (float(distance_cells_to_base.mean()) / self.speed_command
                                      + self.speed_command / (2 * self.BASE_APPROACH_DECELERATION))

        ##### Global map. The world builds it on the first registration, every drone then holds the #####
        ##### same arrays: a camera writing in its own map writes in everybody's.                    #####
        self.world = self._config["world"]
        if self.world is None:
            raise ValueError("the centralized drone needs the CentralizedWorld of the simulation, see main.py")
        self.world.register(self)
        self.map = self.world.map
        self.is_cell_visited = self.world.is_cell_visited
        self.total_uncertainty = self.map[:,:,0].sum()
        self.accomulated_uncertainty = 0.0
        ##### Cells seen by this drone's own camera, to tell what each drone mapped: is_cell_visited #####
        ##### is the shared map, so it holds what the whole swarm saw.                                #####
        self.own_cells_seen = np.zeros((self.MAP_WIDTH, self.MAP_HEIGHT), dtype=bool)

        ### It's considered that the at any high the camera reach will be enough #####
        ##### Cluster plugins initialization #####
        self.fitness = FitnessEvaluator(map_width=self.MAP_WIDTH,
                                        map_height=self.MAP_HEIGHT,
                                        distance_between_cells = self.DISTANCE_BETWEEN_CELLS,
                                        camera_angle=self.CAMERA_ANGLE,
                                        base_variance=self.BASE_VARIANCE,
                                        energy_gamma=self.ENERGY_GAMMA,
                                        alpha_variance_modifier=self.ALPHA_VARIANCE_MODIFIER,
                                        distance_between_drone_norm=self.DISTANCE_BETWEEN_DRONE_NORM,
                                        kernel_n_sigma=self.KERNEL_N_SIGMA,
                                        discharge_rate=self.DISCHARGE_RATE,
                                        ##### The energy factor of the cells keeps the reserve of the scheduler #####
                                        charge_margin=self.MIN_CHARGE_MARGIN,
                                        min_charge_margin=self.MIN_CHARGE_MARGIN,
                                        Nc_charging_base=self.Nc_CHARGING_BASE,
                                        number_of_cells_x_y =self.CELLS_EVALUETED_FOR_PRIORITY,
                                        drone_max_acceleration=self._config["drone_max_acceleration"])

        ##### Pair decisions. At most one with the same drone every TIMEOUT_TO_UPDATE_DESTINATION #####
        self.last_drone_interaction_time = np.zeros(self.NUMBER_OF_DRONES)

        ##### Initial random position #####
        ##### Each drone has its own random streams, one for the initial destination and #####
        ##### one for the wind, derived from (seed, drone id). Drawing from them never   #####
        ##### moves the stream of another drone, so two runs with the same seed keep    #####
        ##### the same scenario for every drone, even after their decisions diverge.     #####
        ##### The same streams as swarm and egocentric: run k gets the same wind there.  #####
        destination_seed, wind_seed = np.random.SeedSequence(
            None if self._config["seed"] is None else [self._config["seed"], self.provider.get_id()]
        ).spawn(2)
        destination_rng = np.random.default_rng(destination_seed)
        self.goto_command = np.array([destination_rng.uniform(-self.DISTANCE_BETWEEN_CELLS*self.MAP_WIDTH/2, self.DISTANCE_BETWEEN_CELLS*self.MAP_WIDTH/2), destination_rng.uniform(-self.DISTANCE_BETWEEN_CELLS*self.MAP_HEIGHT/2, self.DISTANCE_BETWEEN_CELLS*self.MAP_HEIGHT/2), self.DRONE_ALTITUDE])
        ##### No telemetry has arrived yet, but the drones take off from the base, see main.py #####
        command = GotoCoordsMobilityCommand(current_position=(*self.CHARGING_BASE_POSITION, 0.0),
                                            destination=self.goto_command,
                                            speed=self.speed_command)
        self.provider.send_mobility_command(command)

        #### Energy Parameters #####
        self.battery = BatteryPowerPlugin(self, SHARED_BATTERY_CONFIG, rng=np.random.default_rng(wind_seed))
        self.BATTERY_CHECK_INTERVAL = 5.0
        ##### Charge fraction below which the drone is considered lost #####
        self.DEAD_BATTERY_THRESHOLD = 0.10
        self.battery_status = self.battery.battery_status
        # Time this drone reached the base and its battery at that moment. The scheduler books the base with it
        self.charging_start_time = 0.0
        self.battery_at_charging_start = self.battery_status


        ##### Starting the callbacks #####
        self.provider.schedule_timer("mobility",self.provider.current_time() + self.MOBILITY_UPDATE_TIME)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("encounter_check",self.provider.current_time() + self.ENCOUNTER_CHECK_TIME)
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

        ### Updating the cells in the observation range. The map is the global one, every drone sees the change
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
                    self.own_cells_seen[x, y] = True

        self.total_uncertainty = self.map[:,:,0].sum()
        self.accomulated_uncertainty += self.total_uncertainty

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0], [current_x, current_y])

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a accomulated uncertainty of {self.accomulated_uncertainty}")
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")

    ##### The swarm battery terms, with the truth. Nothing in centralized decides with them, they #####
    ##### are kept with the names and signatures of swarm for the recorder and the plots: with     #####
    ##### perfect information the belief of a drone about another is the other's real state.      #####
    @property
    def swarm_battery_status(self) -> np.ndarray:
        """swarm_battery_status of swarm, (N, 3): [battery, time of the observation, status], all true and observed now."""
        return self.world.battery_table(self.provider.current_time())



    ##### Map updating #####
    def vanishing_map_routine(self):
        ##### Grows the global map once per period, whichever drone calls first, see world.grow_uncertainty #####
        self.world.grow_uncertainty(self.provider.current_time())

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, the node {self.provider.get_id()} has {self.MAP_WIDTH*self.MAP_HEIGHT - np.sum(self.is_cell_visited)} unvisited cells")


    def set_destination(self, target_coords) -> None:
        """Flies towards the cell target_coords, given as (row, col)."""
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        target_row, target_col = target_coords
        x_goto = target_row * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        y_goto = target_col * self.DISTANCE_BETWEEN_CELLS - map_center_offset
        self.goto_command = np.array([x_goto, y_goto, self.DRONE_ALTITUDE])
        command = GotoCoordsMobilityCommand(current_position=self.drone_position,
                                            destination=self.goto_command,
                                            speed=self.speed_command)
        self.provider.send_mobility_command(command)

    def horizontal_velocity(self):
        """[vx, vy] of the last telemetry, or None before the first one (then the turn time is 0)."""
        if self.drone_velocity is None:
            return None
        return [float(self.drone_velocity[0]), float(self.drone_velocity[1])]

    ##### Self mobility command. When the drone reaches the destination, it calculates the next one #####
    def internal_mobility_command(self):
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2

        target_coords, value, command = self.fitness.choose_one_cell(
            self.map[:, :, 0],
            self.drone_position,
            self.CHARGING_BASE_POSITION,
            self.battery.battery_status,
            self.speed_command,
            map_center_offset=map_center_offset,
            drone_velocity=self.horizontal_velocity(),
        )
        self.status = DroneStatus.MAPPING

        if self.VERBOSE:
            self._log.info(f"Drone {self.provider.get_id()} going to cell ({target_coords[0]}, {target_coords[1]})")

        self.set_destination(target_coords)

    def go_to_base(self):
        """
        Sent by the scheduler. Same destination as a swarm drone going to the base: the base
        cell, which is the base itself (the bases are snapped to cell centres).
        """
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2
        self.status = DroneStatus.GOING_TO_BASE

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} is sent to the base "
                           f"by the scheduler, battery {self.battery.battery_status:.3f}")

        self.set_destination(self.fitness.position_to_cell(self.CHARGING_BASE_POSITION, map_center_offset))

    ##### Pair decision, the encounter of swarm. Both drones are MAPPING, the one with the highest ID decides #####
    def external_mobility_command(self, another_drone: "Drone") -> None:
        map_center_offset = (self.MAP_WIDTH * self.DISTANCE_BETWEEN_CELLS) / 2

        destinations, best_fitness, commands = self.fitness.choose_two_cells(
            self.map[:, :, 0],
            self.drone_position,
            tuple(another_drone.drone_position),
            self.CHARGING_BASE_POSITION,
            self.battery.battery_status,
            another_drone.battery.battery_status,
            self.speed_command,
            map_center_offset=map_center_offset,
            ##### The planner sees every drone, so the other drone's velocity is read directly #####
            first_drone_velocity=self.horizontal_velocity(),
            second_drone_velocity=another_drone.horizontal_velocity(),
        )

        if self.VERBOSE:
            self._log.info(f"Drone {self.provider.get_id()} going to cell {destinations[0]} and sending drone "
                           f"{another_drone.provider.get_id()} to cell {destinations[1]}.")

        self.set_destination(destinations[0])
        another_drone.receive_destination(destinations[1])

    def receive_destination(self, target_coords) -> None:
        """The destination chosen for this drone by the other drone of a pair decision, the goto message of swarm."""
        self.set_destination(target_coords)

    def encounter_routine(self):
        """
        Pair decisions with every other MAPPING drone within ENCOUNTER_RADIUS whose id is lower
        (the drone with the highest id decides, as in swarm), at most one per pair every
        TIMEOUT_TO_UPDATE_DESTINATION. The distance is the 3D one, as gradysim's range.
        """
        if self.drone_position is None:
            return
        now = self.provider.current_time()
        my_id = self.provider.get_id()
        for other in self.world.others(my_id):
            other_id = other.provider.get_id()
            if other_id > my_id or other.status != DroneStatus.MAPPING or other.drone_position is None:
                continue
            if now - self.last_drone_interaction_time[other_id] <= self.TIMEOUT_TO_UPDATE_DESTINATION:
                continue
            distance = np.linalg.norm(np.array(self.drone_position) - np.array(other.drone_position))
            if distance > self.ENCOUNTER_RADIUS:
                continue

            Drone.Number_of_Encounters += 1
            self.external_mobility_command(other)
            self.last_drone_interaction_time[other_id] = now

    def start_charging(self) -> None:
        """
        The drone is on a pad: it charges for TIME_TO_RECHARGE, then "recharge_done" takes it
        off. Called when the base gives it a pad (PAD_MESSAGE): at once on a free pad, or when
        its turn in the waiting room comes.
        """
        self.status = DroneStatus.CHARGING
        self.charging_start_time = self.provider.current_time()
        self.battery_at_charging_start = self.battery.battery_status

        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} is on a pad and is recharging for {self.TIME_TO_RECHARGE}")

        self.provider.schedule_timer("recharge_done", self.provider.current_time() + self.TIME_TO_RECHARGE)

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

        ##### Handled outside of the status gate below, the drone is CHARGING when it fires #####
        if timer == "recharge_done":
            ##### The drone takes off: it tells the base, which gives its pad to the first drone #####
            ##### of the waiting room, and the battery plugin bills the flight again.             #####
            send_take_off(self.provider, self.BASE_NODE_ID)
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
            self.provider.schedule_timer("encounter_check", self.provider.current_time() + self.ENCOUNTER_CHECK_TIME)
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
                        ##### The scheduler is asked first: when the drone's time to leave has come it #####
                        ##### goes to the base, wherever it was flying to.                              #####
                        if self.world.time_to_leave(self.provider.get_id(), self.provider.current_time()):
                            self.go_to_base()

                        ##### The tolerance has to cover the distance flown between two #####
                        ##### ticks, otherwise the drone passes the destination without #####
                        ##### ever reporting the arrival.                               #####
                        elif distance_to_goto < self.speed_command * self.MOBILITY_UPDATE_TIME:
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
                            ##### routine (camera, encounter check, mobility, traveled distance,       #####
                            ##### battery check) stops until "recharge_done". On the ground the motors  #####
                            ##### are off: the battery plugin stops, so the battery does not drain      #####
                            ##### while the drone waits or charges, and it cannot die there.            #####
                            self.provider.send_mobility_command(SetVelocityMobilityCommand(0.0, 0.0, 0.0))
                            self.battery.shutdown()

                            ##### It waits for a pad until the base gives it one (PAD_MESSAGE): at the  #####
                            ##### same instant when one is free, otherwise when its turn comes. At most #####
                            ##### Nc drones charge at once, see recharge_base.py.                       #####
                            self.status = DroneStatus.WAITING
                            if self.VERBOSE:
                                self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} reached the base and asks for a pad")
                            send_landed(self.provider, self.BASE_NODE_ID, self.battery.battery_status)
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

            ##### The heartbeat of swarm. No message: the positions are read from the world #####
            if timer == "encounter_check":
                if self.status == DroneStatus.MAPPING:
                    self.encounter_routine()
                self.provider.schedule_timer("encounter_check", self.provider.current_time() + self.ENCOUNTER_CHECK_TIME)

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

                    ##### No timer is rescheduled from here on, so every routine stops. The #####
                    ##### others know it is DEAD from the world, no notification is needed. #####
                    return

                self.provider.schedule_timer("battery_check", self.provider.current_time() + self.BATTERY_CHECK_INTERVAL)



    def handle_packet(self, message: str) -> None:
        ##### The only messages of centralized are the ones with the base (recharge_base.py): a #####
        ##### drone receives the PAD its base gives it, nothing else is ever sent to a drone     #####
        if json.loads(message).get('message_type') == BaseMessageType.PAD_MESSAGE.value:
            if self.status == DroneStatus.WAITING:
                self.start_charging()
            return
        self._log.warning(f"Centralized drone {self.provider.get_id()} received a message, only the PAD of the base is expected: {message[:200]}")

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
                "drone_status": int(self.status.value),
                ##### Centralized only: the map is shared, so unvisited_cells is the swarm's. This #####
                ##### is what this drone's own camera saw.                                          #####
                "own_cells_seen": int(self.own_cells_seen.sum()),
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
    distance_between_drone_norm: float,  # tunable
    kernel_n_sigma: int,                 # fixed
    min_charge_margin: float,            # fixed, the reserve of the scheduler
    Nc_charging_base: int,               # fixed
    discharge_rate: float,               # fixed
    charging_base_position: tuple,       # fixed
    results_aggregator: dict,
    world,                               # the CentralizedWorld of the simulation
    encounter_radius: float = 100.0,     # fixed, m
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

    world is the CentralizedWorld every drone of the simulation shares: a new one
    for every simulation, main.create_and_run_simulation builds it.
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
        "distance_between_drone_norm": distance_between_drone_norm,
        "kernel_n_sigma": kernel_n_sigma,
        "min_charge_margin": min_charge_margin,
        'Nc_charging_base': Nc_charging_base,
        "discharge_rate": discharge_rate,
        "charging_base_position": charging_base_position,
        "results_aggregator": results_aggregator,
        "world": world,
        "encounter_radius": encounter_radius,
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
