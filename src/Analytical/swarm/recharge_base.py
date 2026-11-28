"""
The base owns its Nc charging pads and its waiting room: at most Nc drones charge at
once. A drone that lands while every pad is taken waits on the ground next to the base,
in the waiting room, and takes the first pad that frees, in the order the drones landed.

On the ground, waiting or charging, the motors are off: the drone stops its battery
plugin when it lands and starts it again when it takes off, so its battery does not change
and no drone can die while it waits.

The drones talk to their base with messages:

    drone lands      ── LANDED (its battery) ──▶  base: a pad free and nobody waiting?
                     ◀── PAD ─────────────────    yes: at once. No: when its turn comes
    drone charges charge_duration(battery) on the pad
    drone takes off  ── TAKE_OFF ─────────────▶  base: the pad goes to the first drone of
                                                  the waiting room, with a PAD, same instant

A landed drone is WAITING until the PAD arrives. On a free pad the PAD comes at the
same instant so the drone is WAITING for 0 s. The drone and its base are at the same point,
so these messages are always in range. The medium of the runs has no failures; with
failures a drone could wait forever, because there is no check messages implemented yet

The base also tells the drones in range who is on the ground, every HEARTBEAT_INTERVAL:
the drones charging (id, time their charge started, battery) and the drones of the
waiting room (id, time their charge will start, battery), see BaseHeartbeatMessage.
The base computes that start (waiting_room_starts): it is the one that hands the pads
over, so the rule of the line is written only here, and the drones store the start as
they get it. For that the base knows how long each charge lasts, charge_duration of the
battery it starts with, with RECHARGE_FIXED_TIME and RECHARGE_TIME_PER_BATTERY of main.py,
the same values main.py gives the drones.

Only the base node writes its pads and its waiting room. They are made when the
simulation is built (instantiate), before any node is initialized, and main.py hands the
node itself to what reads it after the build
(simulation.get_node(base_node_id(...)).protocol_encapsulator.protocol): the reports of
main_test.py (global_map.recharge_base). Until 2026-11-25 the pads were a
ChargingStation object made in main.py and shared with the node, in
Analytical/charging_station.py.

The base node is added after the drones, so the drones keep the ids
0..number_of_drones-1 the protocols index their arrays with: the base is node
number_of_drones, see base_node_id.
"""

import enum
import json
import heapq
import logging
from collections import deque
from typing import Type, TypedDict

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand, SendMessageCommand


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
    base: int  # index of the base in charging_base_positions
    pads: int  # its number of pads
    ##### [drone id, time its charge started, battery then] of every drone on a pad, the charge #####
    ##### ends charge_duration(battery) later                                                    #####
    charging: list
    ##### [drone id, time its charge will start, battery then] of its waiting room, first in line #####
    ##### first. The base computes the start, see RechargeBase.waiting_room_starts                #####
    waiting: list


def base_node_id(number_of_drones: int, base_index: int = 0) -> int:
    """Node id of a base: the bases are added after the drones"""
    return number_of_drones + base_index


def charge_duration(battery: float, fixed_time: float, time_per_battery: float) -> float:
    """
    How long a charge lasts on a pad: fixed_time, plus time_per_battery in proportion to the
    battery missing when it starts, fixed_time + (1 - battery) * time_per_battery (main.py:
    100 + (1 - battery) * 300 s). The battery does not change on the ground, so the battery at
    landing gives the same duration.
    """
    missing = 1.0 - min(1.0, max(0.0, float(battery)))
    return float(fixed_time) + missing * float(time_per_battery)


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
    """
    The base node: its pads and its waiting room, the landings and the take-offs. Only the
    node writes them, the reports read them from the node, see the module docstring.
    """
    _log: logging.Logger

    _config = {
        "base_index": 0,  # position of this base in charging_base_positions
        "pads": 1,  # how many drones charge on it at once
        ##### A charge lasts charge_duration of its battery: RECHARGE_FIXED_TIME and RECHARGE_TIME_PER_BATTERY of #####
        ##### main.py, the same the drones get                                                                    #####
        "recharge_fixed_time": 100.0,
        "recharge_time_per_battery": 300.0,
        "broadcast": True,  # send the BaseHeartbeatMessage
        "mode": "train",
    }

    @classmethod
    def instantiate(cls, provider) -> "RechargeBase":
        """
        The pads and the waiting room are made here, when the simulation is built, and not in
        initialize: gradysim initializes the nodes in id order, the drones before the base, so
        whatever reads the base before its own initialize already finds them.
        """
        base = super().instantiate(provider)
        base.BASE_INDEX = cls._config["base_index"]
        ##### Nc is int(drones / 3): below 3 drones the base still has one pad #####
        base.pads = max(1, int(cls._config["pads"]))
        base.RECHARGE_FIXED_TIME = float(cls._config["recharge_fixed_time"])
        base.RECHARGE_TIME_PER_BATTERY = float(cls._config["recharge_time_per_battery"])
        ##### {drone id: (time its charge started, its battery then)} of the drones on a pad, #####
        ##### in the order they took it                                                       #####
        base.charging = {}
        base.waiting = deque()    # ids of the drones in the waiting room, the first landed first
        ##### {drone id: (landing time, battery at landing)} of every drone on the ground. The #####
        ##### battery does not change on the ground, so it is also the one its charge starts with #####
        base._landed = {}
        ##### (drone id, time it landed, time it took a pad) of every drone that had to wait #####
        base.waits = []
        base.landings = 0
        return base

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.VERBOSE = self._config.get("mode", "train") == "test"
        self.BROADCAST = self._config["broadcast"]

        ##### Same period as the ChargingNotificationMessage it replaces #####
        self.HEARTBEAT_INTERVAL = 1.0
        if self.BROADCAST:
            self.provider.schedule_timer("base_heartbeat", self.provider.current_time() + self.HEARTBEAT_INTERVAL)

    ##### The pads and the waiting room 

    def land(self, drone_id: int, now: float, battery: float) -> bool:
        """
        The drone landed on the base. True: it takes a pad now. False: it goes to the
        waiting room, because every pad is taken or other drones are already waiting.
        """
        if drone_id in self._landed:
            ##### Already on the ground: the base answers the same again 
            return drone_id in self.charging
        self.landings += 1
        self._landed[drone_id] = (float(now), float(battery))
        if not self.waiting and len(self.charging) < self.pads:
            self.charging[drone_id] = (float(now), float(battery))
            return True
        self.waiting.append(drone_id)
        return False

    def take_off(self, drone_id: int, now: float) -> list:
        """
        The drone finished charging and took off: its pad is free. Returns the drones of
        the waiting room that take a pad now, first in line first (one at most, unless a
        pad was already free).
        """
        self.charging.pop(drone_id, None)
        self._landed.pop(drone_id, None)
        granted = []
        while self.waiting and len(self.charging) < self.pads:
            next_drone = self.waiting.popleft()
            landed_at, battery = self._landed[next_drone]
            self.charging[next_drone] = (float(now), battery)
            self.waits.append((next_drone, landed_at, float(now)))
            granted.append(next_drone)
        return granted

    def charge_duration(self, battery: float) -> float:
        """How long a charge started with this battery lasts, see the module charge_duration."""
        return charge_duration(battery, self.RECHARGE_FIXED_TIME, self.RECHARGE_TIME_PER_BATTERY)

    def waiting_room_starts(self, now: float) -> list:
        """
        When the drones of the waiting room start charging, first in line first. Every charge
        lasts charge_duration of its battery and the base hands a pad over the instant it frees, so the
        k-th drone in line takes the k-th pad that frees: at the end of a charge on a pad,
        or of a charge of the waiting room before it. A pad already free is taken now.

        Exact as long as no drone lands ahead of the line, and none can (first landed, first
        served, see land). The base sends it in its heartbeat, so the drones get the
        start already made and the rule of the line is only written here.
        """
        if not self.waiting:
            return []
        pads_free_at = [start + self.charge_duration(battery) for start, battery in self.charging.values()]
        pads_free_at += [float(now)] * max(0, self.pads - len(pads_free_at))
        heapq.heapify(pads_free_at)
        starts = []
        for drone_id in self.waiting:
            start = max(float(now), heapq.heappop(pads_free_at))
            starts.append(start)
            heapq.heappush(pads_free_at, start + self.charge_duration(self._landed[drone_id][1]))
        return starts

    def charging_rows(self) -> list:
        """[drone id, time its charge started, battery then] of every drone on a pad."""
        return [[drone_id, start, battery] for drone_id, (start, battery) in self.charging.items()]

    def waiting_rows(self, now: float) -> list:
        """[drone id, time its charge will start, battery then] of every drone of the waiting room, first in line first."""
        return [[drone_id, start, self._landed[drone_id][1]]
                for drone_id, start in zip(self.waiting, self.waiting_room_starts(now))]

    def summary(self, end_time: float = None) -> dict:
        """
        Waits of the run, for the reports. With end_time, the drones still waiting then
        count with the time they had waited so far.
        """
        durations = [took - landed for _, landed, took in self.waits]
        if end_time is not None:
            durations += [float(end_time) - self._landed[drone_id][0] for drone_id in self.waiting]
        return {
            "pads": self.pads,
            "landings": self.landings,
            "waited": len(durations),
            "total_wait": float(sum(durations)),
            "max_wait": float(max(durations)) if durations else 0.0,
            "still_waiting": len(self.waiting),
        }

    def report(self, end_time: float, charging_counts=None) -> str:
        """The lines of the "Base" section of the main_test reports."""
        summary = self.summary(end_time)
        share = 100.0 * summary["waited"] / summary["landings"] if summary["landings"] else 0.0
        lines = [f"Pads: {summary['pads']}. Landings: {summary['landings']}, "
                 f"{summary['waited']} of them on a full base ({share:.0f}%)"]
        if summary["waited"]:
            lines.append(f"Waiting room: {summary['total_wait']:.0f} drone*s in total, "
                         f"{summary['total_wait'] / summary['waited']:.0f} s per wait on average, "
                         f"{summary['max_wait']:.0f} s at most"
                         + (f", {summary['still_waiting']} still waiting at the end"
                            if summary["still_waiting"] else ""))
        if charging_counts is not None and len(charging_counts):
            lines.append(f"Most drones charging at once: {int(max(charging_counts))}")
        return "\n".join(lines)

    ##### Messages #####

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
            'pads': self.pads,
            'charging': self.charging_rows(),
            'waiting': self.waiting_rows(self.provider.current_time()),
        }
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(message)))

    def handle_timer(self, timer: str) -> None:
        if timer == "base_heartbeat":
            ##### Nothing to tell while the base is empty #####
            if self.charging or self.waiting:
                self.send_heartbeat()
            self.provider.schedule_timer("base_heartbeat", self.provider.current_time() + self.HEARTBEAT_INTERVAL)

    def handle_packet(self, message: str) -> None:
        data: dict = json.loads(message)
        msg_type = data.get('message_type')
        now = self.provider.current_time()

        if msg_type == BaseMessageType.LANDED_MESSAGE.value:
            landed: LandedMessage = data
            # If there is an available pad send it to the drone. It will be charging
            if self.land(landed['sender'], now, landed['battery']):
                self.send_pad(landed['sender'])
            # else, it will wait in the "waiting room"
            elif self.VERBOSE:
                self._log.info(f"At time: {now}, node {landed['sender']} landed on base {self.BASE_INDEX}, "
                               f"every pad is taken: waiting room, {len(self.waiting)} in line")

        elif msg_type == BaseMessageType.TAKE_OFF_MESSAGE.value:
            take_off: TakeOffMessage = data
            # Call the waiting drones to the charging pad, if there is any
            for drone_id in self.take_off(take_off['sender'], now):
                self.send_pad(drone_id)

        ##### Anything else is a broadcast of the drones (heartbeats, dead notifications), #####
        ##### sent to everyone in range, bases included. None of them is meant for a base. #####

    def handle_telemetry(self, telemetry) -> None:
        ##### The mobility handler sends telemetry to every node, the base never moves #####
        pass

    def finish(self) -> None:
        if self.VERBOSE:
            self._log.info(f"Base {self.BASE_INDEX} (node {self.provider.get_id()}) finished with "
                           f"{len(self.charging)} drones charging and {len(self.waiting)} waiting")


def recharge_base_factory(pads: int, recharge_fixed_time: float, recharge_time_per_battery: float, base_index: int = 0, broadcast: bool = True,
                          mode: str = "train") -> Type[RechargeBase]:
    """
    Creates the protocol class of the base, with pads charging pads (Nc), new for every
    simulation. recharge_fixed_time and recharge_time_per_battery set how long a charge lasts (charge_duration), the values main.py
    also gives the drones. broadcast turns on the BaseHeartbeatMessage.
    """
    config = {
        "base_index": base_index,
        "pads": pads,
        "recharge_fixed_time": recharge_fixed_time,
        "recharge_time_per_battery": recharge_time_per_battery,
        "broadcast": broadcast,
        "mode": mode,
    }

    class ConfiguredRechargeBase(RechargeBase):
        _config = config

    return ConfiguredRechargeBase
