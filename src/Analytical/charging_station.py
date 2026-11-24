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

One station is built per base and per simulation (main.py) and given to every drone
through the protocol factory, the way results_aggregator is. It is the bookkeeping of
the base itself: which pads are free, who landed first. The drones do not decide with
it. A drone only learns that it has to wait once it lands, and swarm still estimates the
congestion from what it hears (aproximate_battery_consumption_model).

A waiting drone asks for a pad once per protocol tick (take_pad), so a freed pad can stay
empty for up to that tick before the first drone in line takes it.
"""

from collections import deque


class ChargingStation:
    def __init__(self, pads: int):
        ##### Nc is int(drones / 3): below 3 drones the base still has one pad #####
        self.pads = max(1, int(pads))
        self.charging = []        # ids of the drones on a pad
        self.waiting = deque()    # ids of the drones in the waiting room, the first landed first
        self._landed_at = {}      # landing time of every drone on the ground
        ##### (drone id, time it landed, time it took a pad) of every drone that had to wait #####
        self.waits = []
        self.landings = 0

    def land(self, drone_id: int, now: float) -> bool:
        """
        The drone landed on the base. True: it takes a pad now. False: it goes to the
        waiting room, because every pad is taken or other drones are already waiting.
        """
        self.landings += 1
        self._landed_at[drone_id] = float(now)
        if not self.waiting and len(self.charging) < self.pads:
            self.charging.append(drone_id)
            return True
        self.waiting.append(drone_id)
        return False

    def take_pad(self, drone_id: int, now: float) -> bool:
        """A drone of the waiting room asks for a pad. It gets one when a pad is free and it is first in line."""
        if self.waiting and self.waiting[0] == drone_id and len(self.charging) < self.pads:
            self.waiting.popleft()
            self.charging.append(drone_id)
            self.waits.append((drone_id, self._landed_at[drone_id], float(now)))
            return True
        return False

    def take_off(self, drone_id: int) -> None:
        """The drone finished charging and took off: its pad is free."""
        self.charging.remove(drone_id)
        self._landed_at.pop(drone_id, None)

    def summary(self, end_time: float = None) -> dict:
        """
        Waits of the run, for the reports. With end_time, the drones still waiting then
        count with the time they had waited so far.
        """
        durations = [took - landed for _, landed, took in self.waits]
        if end_time is not None:
            durations += [float(end_time) - self._landed_at[drone_id] for drone_id in self.waiting]
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
