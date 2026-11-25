"""
The charging pads of a recharge base, with a waiting room (since 2026-11-24).

The base has `pads` charging pads, so at most that many drones charge at once (Nc). A
drone that lands while every pad is taken waits on the ground next to the base, in the
waiting room, and takes the first pad that frees, in the order the drones landed.

A drone on the ground, waiting or charging, has its motors off: the protocol stops its
battery plugin when it lands and starts it again when it takes off. Its battery does not
change on the ground, so no drone can die while it waits.

Before this, the capacity was only a penalty of the cost: any number of drones could
charge at once, so a system that ignored Nc got its drones back to mapping sooner and
looked better on uncertainty than one that respected it. Now waiting costs mapping time,
for every system.

This is the bookkeeping of the base node (recharge_base.RechargeBase, since 2026-11-24),
and only the base writes to it: a drone tells the base it landed or took off with a
message, and the base answers with the pad. The base hands a freed pad to the first
drone of the waiting room at the instant the other one takes off, so a pad is never
empty while a drone waits. main.py keeps a reference to it for the reports, and the
centralized world reads it, as the oracle it is.
"""

import heapq
from collections import deque


class ChargingStation:
    def __init__(self, pads: int):
        ##### Nc is int(drones / 3): below 3 drones the base still has one pad #####
        self.pads = max(1, int(pads))
        ##### {drone id: (time its charge started, its battery then)} of the drones on a pad, #####
        ##### in the order they took it                                                       #####
        self.charging = {}
        self.waiting = deque()    # ids of the drones in the waiting room, the first landed first
        ##### {drone id: (landing time, battery at landing)} of every drone on the ground. The #####
        ##### battery does not change on the ground, so it is also the one its charge starts with #####
        self._landed = {}
        ##### (drone id, time it landed, time it took a pad) of every drone that had to wait #####
        self.waits = []
        self.landings = 0

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

    def as_message(self) -> dict:
        """What the base broadcasts about this station, see recharge_base.BaseHeartbeatMessage."""
        return {"pads": self.pads, "charging": self.charging_rows(), "waiting": self.waiting_rows()}

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


def waiting_room_starts(charging_starts, waiting_count: int, pads: int, time_to_recharge: float,
                        now: float) -> list:
    """
    When the drones of a waiting room start charging, first in line first. Every charge
    lasts time_to_recharge and the base hands a pad over the instant it frees, so the
    k-th drone in line takes the k-th pad that frees: at the end of a charge on the base,
    or of a charge of the waiting room before it. A pad already free is taken now.

    charging_starts are the start times of the charges on the pads. Exact as long as no
    drone lands ahead of the line, and none can (first landed, first served). Used by the
    swarm drones on the base heartbeat and by the centralized scheduler.
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
