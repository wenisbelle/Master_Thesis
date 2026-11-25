"""
The recharge bases of the expanding_testing swarm system, each one a node of the
simulation (since 2026-11-24). Every system has its own copy of this file (since
2026-11-25), so each one runs on its own.

Every base owns its pads and its waiting room, and manages the landings on it alone, as
a real base would: at most `pads` drones charge on it at once (PADS_PER_BASE in main.py,
Nc by default). A drone that lands on it while every pad is taken waits on the ground
next to it, in its waiting room, and takes the first of its pads that frees, in the
order the drones landed on it.

On the ground, waiting or charging, the motors are off: the drone stops its battery
plugin when it lands and starts it again when it takes off, so its battery does not change
and no drone can die while it waits. Before the pads (2026-11-24) the capacity was only a
penalty of the cost: any number of drones could charge at once, so a system that ignored
Nc got its drones back to mapping sooner and looked better on uncertainty than one that
respected it. Now waiting costs mapping time, for every system.

The drones talk to their base with messages:

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

The base also tells the drones in range who is on the ground, every HEARTBEAT_INTERVAL:
the drones charging (id, time their charge started, battery) and the drones of the
waiting room (id, time they landed, battery), see BaseHeartbeatMessage. It replaces the
ChargingNotificationMessage each charging drone broadcast itself before 2026-11-24. The
base is at the drones' altitude, so it reaches as far as that notification did. It is
sent only while some drone is on the ground, as the notification was.

A base tells only about itself. A drone learns about the other bases from the drones it
meets, which pass on what they heard (the rows of the battery table, newest wins). Until
2026-11-25 the bases shared the list of every base (a "second radio" between them) and
each heartbeat told about all of them.

Only the base node writes its pads and its waiting room. They are made when the
simulation is built (instantiate), before any node is initialized, and main.py hands the
node itself to what reads it after the build
(simulation.get_node(base_node_id(...)).protocol_encapsulator.protocol): the reports of
main_test.py (global_map.recharge_bases, one node per base). Until 2026-11-25 the pads
were a ChargingStation object made in main.py and shared with the node, one per base, in
Analytical/charging_station.py.

The base nodes are added after the drones, so the drones keep the ids
0..number_of_drones-1 the protocols index their arrays with: base b is node
number_of_drones + b, see base_node_id.
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
    ##### ends TIME_TO_RECHARGE later                                                            #####
    charging: list
    waiting: list  # [drone id, time it landed, battery then] of its waiting room, first in line first


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


def waiting_room_starts(charging_starts, waiting_count: int, pads: int, time_to_recharge: float,
                        now: float) -> list:
    """
    When the drones of a waiting room start charging, first in line first. Every charge
    lasts time_to_recharge and the base hands a pad over the instant it frees, so the
    k-th drone in line takes the k-th pad that frees: at the end of a charge on the base,
    or of a charge of the waiting room before it. A pad already free is taken now.

    charging_starts are the start times of the charges on the pads. Exact as long as no
    drone lands ahead of the line, and none can (first landed, first served). Used by
    the swarm drones on the base heartbeat.
    """
    if waiting_count <= 0:
        return []
    pads_free_at = [float(start) + time_to_recharge for start in charging_starts]
    pads_free_at += [float(now)] * max(0, int(pads) - len(pads_free_at))
    heapq.heapify(pads_free_at)
    starts = []
    for _ in range(waiting_count):
        start = max(float(now), heapq.heappop(pads_free_at))
        starts.append(start)
        heapq.heappush(pads_free_at, start + time_to_recharge)
    return starts


class RechargeBase(IProtocol):
    """
    The base node: its pads and its waiting room, the landings and the take-offs. Only the
    node writes them, the reports read them from the node, see the module docstring.
    """
    _log: logging.Logger

    _config = {
        "base_index": 0,  # position of this base in charging_base_positions
        "pads": 1,  # how many drones charge on it at once
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
        ##### PADS_PER_BASE of main.py, Nc by default. At least 1 #####
        base.pads = max(1, int(cls._config["pads"]))
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

    ##### The pads and the waiting room #####

    def land(self, drone_id: int, now: float, battery: float) -> bool:
        """
        The drone landed on the base. True: it takes a pad now. False: it goes to the
        waiting room, because every pad is taken or other drones are already waiting.
        """
        if drone_id in self._landed:
            ##### Already on the ground: the base answers the same again #####
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

    def charging_rows(self) -> list:
        """[drone id, time its charge started, battery then] of every drone on a pad."""
        return [[drone_id, start, battery] for drone_id, (start, battery) in self.charging.items()]

    def waiting_rows(self) -> list:
        """[drone id, landing time, battery then] of every drone of the waiting room, first in line first."""
        return [[drone_id, *self._landed[drone_id]] for drone_id in self.waiting]

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
            'waiting': self.waiting_rows(),
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
            if self.land(landed['sender'], now, landed['battery']):
                self.send_pad(landed['sender'])
            elif self.VERBOSE:
                self._log.info(f"At time: {now}, node {landed['sender']} landed on base {self.BASE_INDEX}, "
                               f"every pad is taken: waiting room, {len(self.waiting)} in line")

        elif msg_type == BaseMessageType.TAKE_OFF_MESSAGE.value:
            take_off: TakeOffMessage = data
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


def recharge_base_factory(pads: int, base_index: int = 0, broadcast: bool = True,
                          mode: str = "train") -> Type[RechargeBase]:
    """
    Creates the protocol class of the base at charging_base_positions[base_index], with pads
    charging pads, new for every simulation. broadcast turns on the BaseHeartbeatMessage.
    """
    config = {
        "base_index": base_index,
        "pads": pads,
        "broadcast": broadcast,
        "mode": mode,
    }

    class ConfiguredRechargeBase(RechargeBase):
        _config = config

    return ConfiguredRechargeBase
