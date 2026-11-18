"""
Recharge base of the multi-base scenario. A base is a node of the simulation,
like a drone, but it never moves.

A drone that reaches a base plugs into it, so the base always knows exactly
which drones are charging on it. The bases of a simulation also know exactly
which drones are charging on every other base: in reality that would be a
second radio, long range and low bandwidth, here it is the ChargingRegistry
every base of the simulation shares.

The egocentric drones only look at their own battery, so here the bases send
nothing: they are the same nodes as in the swarm system, at the same places,
without its heartbeat. The registry is still kept, the drones plug into it the
same way.
"""

import logging
from typing import Type

from gradysim.protocol.interface import IProtocol


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

    def handle_timer(self, timer: str) -> None:
        pass

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
