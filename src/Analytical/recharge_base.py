"""
The recharge base, a node of the simulation (since 2026-11-24), shared by every
Analytical system and by the expanding_testing copies.

Before, the pads of the base were a Python object every drone called directly
(charging_station.py). Now the base is a gradysim node, like a drone that never
moves, and it owns its pads: the drones talk to it with messages.

    drone lands      ── LANDED (its battery) ──▶  base: a pad free and nobody waiting?
                     ◀── PAD ─────────────────    yes: at once. No: when its turn comes
    drone charges TIME_TO_RECHARGE on the pad
    drone takes off  ── TAKE_OFF ─────────────▶  base: the pad goes to the first drone of
                                                  the waiting room, with a PAD, same instant

A landed drone is WAITING until the PAD arrives. On a free pad the PAD comes at the
same instant (gradysim delivers a message with no delay as a new event of the same
time), so the drone is WAITING for 0 s. The drone and its base are at the same point,
so these messages are always in range. The medium of the runs has no failures; with
failures a drone could wait forever.

With broadcast on (the swarm system), the base also tells the drones in range who is
on the ground, every HEARTBEAT_INTERVAL: the drones charging (id, time their charge
started, battery) and the drones of the waiting room (id, time they landed, battery),
see BaseHeartbeatMessage. It replaces the ChargingNotificationMessage each charging
swarm drone broadcast itself before. The base is at the drones' altitude, so it
reaches as far as that notification did. It is sent only while some drone is on the
ground, as the notification was.

Several bases (expanding_testing) share the list of every station of the simulation,
the "second radio" of the bases, long range and low bandwidth: each base writes only
its own station, and broadcasts all of them.

The base nodes are added after the drones, so the drones keep the ids
0..number_of_drones-1 the protocols index their arrays with: base b is node
number_of_drones + b, see base_node_id.
"""

import enum
import json
import logging
from typing import Type, TypedDict

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand, SendMessageCommand

from .charging_station import ChargingStation


##### The protocols fly, and charge, at DRONE_ALTITUDE, and the range of gradysim is 3D: the antenna of the  #####
##### base is put at that altitude too, so its messages reach as far as the drone-to-drone ones. On the       #####
##### ground a base would only reach sqrt(range^2 - 50^2) horizontally, 87 m for a range of 100 m.            #####
BASE_ANTENNA_ALTITUDE = 50.0


class BaseMessageType(enum.Enum):
    ##### Apart from the MessageType values of the drone protocols (0 to 4) #####
    LANDED_MESSAGE = 100
    PAD_MESSAGE = 101
    TAKE_OFF_MESSAGE = 102
    BASE_HEARTBEAT_MESSAGE = 103


class LandedMessage(TypedDict):
    message_type: int
    sender: int  # the drone that landed
    battery: float  # its battery at landing, the same all the time it stays on the ground


class PadMessage(TypedDict):
    message_type: int
    sender: int  # node id of the base
    base: int  # index of the base in charging_base_positions


class TakeOffMessage(TypedDict):
    message_type: int
    sender: int  # the drone that took off, its pad is free


class BaseHeartbeatMessage(TypedDict):
    message_type: int
    sender: int  # node id of the base
    base: int  # index of the sending base in charging_base_positions
    ##### One entry per base of the simulation, in charging_base_positions order (one in the single base   #####
    ##### systems), see ChargingStation.as_message:                                                         #####
    #####   "pads": its number of pads                                                                      #####
    #####   "charging": [drone id, time its charge started, battery then], the charge ends TIME_TO_RECHARGE later #####
    #####   "waiting": [drone id, time it landed, battery then], first in line first                        #####
    bases: list


def base_node_id(number_of_drones: int, base_index: int = 0) -> int:
    """Node id of a base: the bases are added after the drones, see main.py."""
    return number_of_drones + base_index


def send_landed(provider, base_id: int, battery: float) -> None:
    """The drone landed on the base base_id: it asks for a pad."""
    message: LandedMessage = {
        'message_type': BaseMessageType.LANDED_MESSAGE.value,
        'sender': provider.get_id(),
        'battery': float(battery),
    }
    provider.send_communication_command(SendMessageCommand(json.dumps(message), base_id))


def send_take_off(provider, base_id: int) -> None:
    """The drone took off from the base base_id: its pad is free."""
    message: TakeOffMessage = {
        'message_type': BaseMessageType.TAKE_OFF_MESSAGE.value,
        'sender': provider.get_id(),
    }
    provider.send_communication_command(SendMessageCommand(json.dumps(message), base_id))


class RechargeBase(IProtocol):
    _log: logging.Logger

    _config = {
        "base_index": 0,  # position of this base in charging_base_positions
        "stations": None,  # the ChargingStation of every base of the simulation, see the module docstring
        "broadcast": False,  # send the BaseHeartbeatMessage (swarm)
        "mode": "train",
    }

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.VERBOSE = self._config.get("mode", "train") == "test"

        self.BASE_INDEX = self._config["base_index"]
        self.stations = self._config["stations"]
        if not self.stations:
            raise ValueError("the recharge base needs the stations of the simulation, see recharge_base_factory")
        self.station: ChargingStation = self.stations[self.BASE_INDEX]
        self.BROADCAST = self._config["broadcast"]

        ##### Same period as the ChargingNotificationMessage it replaces #####
        self.HEARTBEAT_INTERVAL = 1.0
        if self.BROADCAST:
            self.provider.schedule_timer("base_heartbeat", self.provider.current_time() + self.HEARTBEAT_INTERVAL)

    def send_pad(self, drone_id: int) -> None:
        message: PadMessage = {
            'message_type': BaseMessageType.PAD_MESSAGE.value,
            'sender': self.provider.get_id(),
            'base': self.BASE_INDEX,
        }
        self.provider.send_communication_command(SendMessageCommand(json.dumps(message), drone_id))
        if self.VERBOSE:
            self._log.info(f"At time: {self.provider.current_time()}, base {self.BASE_INDEX} gives a pad to node {drone_id}")

    def send_heartbeat(self) -> None:
        message: BaseHeartbeatMessage = {
            'message_type': BaseMessageType.BASE_HEARTBEAT_MESSAGE.value,
            'sender': self.provider.get_id(),
            'base': self.BASE_INDEX,
            'bases': [station.as_message() for station in self.stations],
        }
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(message)))

    def handle_timer(self, timer: str) -> None:
        if timer == "base_heartbeat":
            ##### Nothing to tell while every base is empty #####
            if any(station.charging or station.waiting for station in self.stations):
                self.send_heartbeat()
            self.provider.schedule_timer("base_heartbeat", self.provider.current_time() + self.HEARTBEAT_INTERVAL)

    def handle_packet(self, message: str) -> None:
        data: dict = json.loads(message)
        msg_type = data.get('message_type')
        now = self.provider.current_time()

        if msg_type == BaseMessageType.LANDED_MESSAGE.value:
            landed: LandedMessage = data
            if self.station.land(landed['sender'], now, landed['battery']):
                self.send_pad(landed['sender'])
            elif self.VERBOSE:
                self._log.info(f"At time: {now}, node {landed['sender']} landed on base {self.BASE_INDEX}, "
                               f"every pad is taken: waiting room, {len(self.station.waiting)} in line")

        elif msg_type == BaseMessageType.TAKE_OFF_MESSAGE.value:
            take_off: TakeOffMessage = data
            for drone_id in self.station.take_off(take_off['sender'], now):
                self.send_pad(drone_id)

        ##### Anything else is a broadcast of the drones (heartbeats, dead notifications), #####
        ##### sent to everyone in range, bases included. None of them is meant for a base. #####

    def handle_telemetry(self, telemetry) -> None:
        ##### The mobility handler sends telemetry to every node, the base never moves #####
        pass

    def finish(self) -> None:
        if self.VERBOSE:
            self._log.info(f"Base {self.BASE_INDEX} (node {self.provider.get_id()}) finished with "
                           f"{len(self.station.charging)} drones charging and {len(self.station.waiting)} waiting")


def recharge_base_factory(stations: list, base_index: int = 0, broadcast: bool = False,
                          mode: str = "train") -> Type[RechargeBase]:
    """
    Creates the protocol class of the base at charging_base_positions[base_index].
    stations is the ChargingStation of every base of the simulation, the same list for
    all of them, new for every simulation. broadcast turns on the BaseHeartbeatMessage.
    """
    config = {
        "base_index": base_index,
        "stations": stations,
        "broadcast": broadcast,
        "mode": mode,
    }

    class ConfiguredRechargeBase(RechargeBase):
        _config = config

    return ConfiguredRechargeBase
