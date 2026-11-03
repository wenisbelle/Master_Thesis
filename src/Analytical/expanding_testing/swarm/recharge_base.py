"""
Recharge base of the multi-base scenario. A base is a node of the simulation,
like a drone, but it never moves.

A drone that reaches a base plugs into it, so the base always knows exactly
which drones are charging on it. The bases of a simulation also know exactly
which drones are charging on every other base: in reality that would be a
second radio, long range and low bandwidth, here it is the ChargingRegistry
every base of the simulation shares.

In the swarm system each base broadcasts that knowledge every
HEARTBEAT_INTERVAL, with the same range as every other message. It replaces
the ChargingNotificationMessage the charging drones used to send themselves:
a drone in range stores the drones charging, and when it learns a new one it
chooses its destination again, see Drone.handle_packet.
"""

import json
import logging
from typing import Type

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand

from .protocol import MessageType, BaseHeartbeatMessage


class ChargingRegistry:
    """
    The drones plugged into each base, shared by every base of one simulation.
    The drones write to it when they plug in and out (Drone.handle_timer), the
    bases read it. A new one is created for every simulation, see main.py.
    """

    def __init__(self, number_of_bases: int):
        ##### Per base, {drone id: (time it started charging, its battery then)} #####
        self.charging = [{} for _ in range(number_of_bases)]

    def dock(self, base_index: int, drone_id: int, charging_start_time: float, battery: float) -> None:
        self.charging[base_index][drone_id] = (float(charging_start_time), float(battery))

    def undock(self, base_index: int, drone_id: int) -> None:
        self.charging[base_index].pop(drone_id, None)

    def counts(self) -> list:
        """Number of drones charging on each base."""
        return [len(drones) for drones in self.charging]

    def as_message(self) -> list:
        """The registry in the format of BaseHeartbeatMessage['charging']."""
        return [[[drone_id, start, battery] for drone_id, (start, battery) in sorted(drones.items())]
                for drones in self.charging]


class RechargeBase(IProtocol):
    _log: logging.Logger

    _config = {
        "base_index": 0,  # position of this base in charging_base_positions
        "registry": None,  # ChargingRegistry shared by every base of the simulation
        "mode": "train",
    }

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.VERBOSE = self._config.get("mode", "train") == "test"

        self.BASE_INDEX = self._config["base_index"]
        self.registry: ChargingRegistry = self._config["registry"]

        ##### Same period as the ChargingNotificationMessage it replaces #####
        self.HEARTBEAT_INTERVAL = 1.0
        self.provider.schedule_timer("base_heartbeat", self.provider.current_time() + self.HEARTBEAT_INTERVAL)

    def send_heartbeat(self) -> None:
        message: BaseHeartbeatMessage = {
            'message_type': MessageType.BASE_HEARTBEAT_MESSAGE.value,
            'sender': self.provider.get_id(),
            'base': self.BASE_INDEX,
            'charging': self.registry.as_message(),
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def handle_timer(self, timer: str) -> None:
        if timer == "base_heartbeat":
            self.send_heartbeat()
            self.provider.schedule_timer("base_heartbeat", self.provider.current_time() + self.HEARTBEAT_INTERVAL)

    def handle_packet(self, message: str) -> None:
        ##### The drones broadcast their heartbeats and dead notifications to everyone #####
        ##### in range, bases included. None of them is meant for a base.              #####
        pass

    def handle_telemetry(self, telemetry) -> None:
        ##### The mobility handler sends telemetry to every node, the base never moves #####
        pass

    def finish(self) -> None:
        if self.VERBOSE:
            self._log.info(f"Base {self.BASE_INDEX} (node {self.provider.get_id()}) finished with "
                           f"{self.registry.counts()[self.BASE_INDEX]} drones charging")


def recharge_base_factory(base_index: int, registry: ChargingRegistry, mode: str = "train") -> Type[RechargeBase]:
    """
    Creates the protocol class of the base at charging_base_positions[base_index].
    Every base of a simulation gets the same registry.
    """
    config = {
        "base_index": base_index,
        "registry": registry,
        "mode": mode,
    }

    class ConfiguredRechargeBase(RechargeBase):
        _config = config

    return ConfiguredRechargeBase
