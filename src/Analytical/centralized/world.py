"""
The central information of the centralized system: what no drone of swarm or
egocentric has.

One CentralizedWorld is built per simulation (main.create_and_run_simulation) and
given to every drone through the protocol factory, the way results_aggregator is.
It is a plain Python object, not a gradysim node: the oracle needs no radio, the
drones read it directly. It also reads the pads and the waiting room of the base
node (recharge_base.py), the truth no drone has.

It holds:
- **The global map.** One map shared by every drone: each drone's `map` and
  `is_cell_visited` are the same arrays. A camera writes into it and every drone
  sees the result at once. The uncertainty grows once per second for the whole map
  (grow_uncertainty), not once per drone.
- **The registry of the drones.** Every drone reads the true position, battery and
  status of the others from it.
- **The charging scheduler** (plan_charging, used by time_to_leave): which drone
  goes to the base, and when.

The scheduler is a rolling-horizon backward greedy. Every time a drone asks, it is
computed again from the true state:
1. The base timeline holds the charges already decided: the drones CHARGING, from
   their arrival, and the drones GOING_TO_BASE, at the slot they left for.
2. Each MAPPING drone has a latest safe arrival: the time at which, leaving for the
   base, it arrives with RESERVE of battery left (MIN_CHARGE_MARGIN of the swarm),
   discharging at PLANNING_DRAIN_FACTOR times DISCHARGE_RATE.
3. Taking the drones from the latest deadline to the earliest, each one gets the
   latest charging slot (TIME_TO_RECHARGE + SLOT_GUARD long) that starts at or before
   its deadline with fewer than the capacity drones charging at the same time. A drone
   that finds none is repaired (place_slots). When there is still none the deadline
   wins: a drone is never sent late to avoid congestion.
4. A drone leaves when the time to start its slot minus its flight time to the base
   has come. It checks once per tick, so by then its slot is a little in the past: a
   free slot that is late by at most SLOT_GUARD means "leave now" (latest_slot).
A full recharge always takes TIME_TO_RECHARGE, whatever the battery left, so a
charge started early only wastes battery: the latest safe slot is the best one, and
the slots only move earlier to keep at most `capacity` drones on the base at once.

The slot problem is the one of Simons (1983): jobs of equal length (the charges),
identical machines (the capacity), release times (the earliest arrivals) and
deadlines. check_centralized.py tests place_slots against an exact solver. The
README has the references and the measurements behind the constants.
"""

from dataclasses import dataclass

import numpy as np

from .recharge_base import waiting_room_starts


@dataclass(frozen=True)
class ChargingSlot:
    """What the scheduler decided for one MAPPING drone, all in simulated seconds."""
    start: float           # the charge starts (the drone reaches the base)
    leave: float           # the drone has to leave its position to start the charge on time
    latest_arrival: float  # the latest arrival that still keeps RESERVE of battery


def on_base(starts, time: float, length: float) -> int:
    """
    Number of drones charging on the base at the instant `time`.

    starts: the start time of each charge. Every charge lasts `length` seconds, so a
    charge that starts at s occupies the base during [s, s + length).
    """
    return sum(1 for other in starts if other <= time < other + length)


def most_on_base(starts, start: float, length: float) -> int:
    """
    Peak number of drones charging at the same time during [start, start + length).
    `starts` and `length` mean the same as in on_base.

    Every charge has the same length, so the count can only go up at the moment a charge
    begins. Checking `start` itself and every charge start inside the interval is enough.
    """
    end = start + length
    return max([on_base(starts, start, length)]
               + [on_base(starts, other, length) for other in starts if start < other < end])


def congestion(starts, start: float, length: float, capacity: int) -> float:
    """
    How many seconds of [start, start + length) the base is already full, meaning
    `capacity` or more of the charges in `starts` are on it. `starts` and `length` mean
    the same as in on_base.
    """
    end = start + length
    points = sorted({start, end} | {t for other in starts for t in (other, other + length) if start < t < end})
    return sum(b - a for a, b in zip(points, points[1:]) if on_base(starts, a, length) >= capacity)


def latest_slot(starts, earliest: float, latest: float, length: float, capacity: int,
                tolerance: float) -> tuple:
    """
    When a drone should start charging. Returns (start, free).
    start: the time, in simulated seconds, when the drone should begin charging.
    start: the time, in simulated seconds, when the drone should begin charging.

    earliest: arrival if it leaves now. latest: its deadline.

    A free slot is a start where the base stays below `capacity` for the whole charge.
    The drone gets the latest free slot in [earliest, latest]. A free slot up to
    `tolerance` before `earliest` also counts, as "leave now".

    If there is none, free is False and the drone gets the start with the least time
    over capacity (latest on ties). If latest <= earliest it simply leaves now.
    """
    if latest <= earliest:
        return earliest, most_on_base(starts, earliest, length) < capacity

    ##### The latest free start is `latest` or ends just as another charge begins #####
    candidates = [latest] + [other - length for other in starts if other - length < latest]
    free = [start for start in candidates if most_on_base(starts, start, length) < capacity]
    if free:
        best = max(free)
        if best >= earliest:
            return best, True
        if best >= earliest - tolerance:
            return earliest, True

    ##### No free slot the drone can reach. The congestion is piecewise linear in the start, #####
    ##### with its corners where an end of the new charge meets an end of another one.       #####
    candidates = {earliest, latest} | {t for other in starts for t in (other - length, other, other + length)
                                       if earliest < t < latest}
    return max(candidates, key=lambda start: (-congestion(starts, start, length, capacity), start)), False


def plan_charging(now: float, mapping_drones, booked_starts, capacity: int, time_to_recharge: float,
                  slot_guard: float, discharge_rate: float, reserve: float) -> dict:
    """
    The charging slot of every MAPPING drone, {drone id: ChargingSlot}.

    mapping_drones: (drone id, true battery, flight time to the base) of every MAPPING drone.
    booked_starts:  the start of every charge already decided (CHARGING and GOING_TO_BASE drones).

    A charge occupies the base for time_to_recharge + slot_guard. The guard absorbs the
    error of the predicted flight time (turning, braking on the approach) and the delay of
    a drone that only checks its plan once per tick, so two charges planned back to back do
    not overlap.

    The drones are taken from the latest deadline to the earliest (ties by id, so the plan
    does not depend on the order they are given in), each one placed at the latest free slot
    before its deadline (latest_slot). This is the backward form of earliest-deadline-first.
    A capacity below 1 (fewer than 3 drones) is planned as 1: every charge is then above it
    anyway, and one at a time is the least.

    A drone with a narrow window (close to its deadline, and far from the base) can find
    every slot it can reach taken by a drone placed before it, that had room to go earlier.
    The greedy alone then misses a schedule within the capacity that exists (8 of 390
    random cases in check_centralized.py). place_slots repairs it, by a swap or by placing
    that drone first.
    """
    deadlines = []
    for drone_id, battery, flight_time in mapping_drones:
        ##### battery - discharge_rate * (wait + flight_time) = reserve, solved for the wait #####
        latest_leave = now + (battery - reserve - discharge_rate * flight_time) / discharge_rate
        deadlines.append((drone_id, latest_leave + flight_time, flight_time))
    return place_slots(now, deadlines, booked_starts, time_to_recharge + slot_guard, max(1, capacity),
                       tolerance=slot_guard)


def _greedy(now: float, order, booked, length: float, capacity: int, tolerance: float, swap: bool):
    """
    One pass of place_slots in the given order. Returns the placements, [drone id, start,
    earliest, latest, flight time] in the order they were placed, and the ids of the
    drones left without a free slot.
    """
    placed, failed = [], []
    for drone_id, latest_arrival, flight_time in order:
        earliest = now + flight_time
        starts = booked + [entry[1] for entry in placed]
        start, free = latest_slot(starts, earliest, latest_arrival, length, capacity, tolerance)

        if not free and swap:
            for k in reversed(range(len(placed))):
                others = booked + [entry[1] for j, entry in enumerate(placed) if j != k]
                new_start, new_free = latest_slot(others, earliest, latest_arrival, length, capacity, tolerance)
                if not new_free:
                    continue
                _, _, moved_earliest, moved_latest, _ = placed[k]
                moved_start, moved_free = latest_slot(others + [new_start], moved_earliest, moved_latest,
                                                      length, capacity, tolerance)
                if moved_free:
                    placed[k][1] = moved_start
                    start, free = new_start, True
                    break

        if not free:
            failed.append(drone_id)
        placed.append([drone_id, start, earliest, latest_arrival, flight_time])
    return placed, failed


def place_slots(now: float, deadlines, booked_starts, length: float, capacity: int,
                tolerance: float, repair: bool = True) -> dict:
    """
    The slot part of plan_charging, given the deadlines: (drone id, latest arrival, flight
    time to the base) of every drone to place. Apart so check_centralized.py can test it
    against an exact solver.

    Each drone gets latest_slot among the charges booked and the ones placed before it,
    from the latest deadline to the earliest. With repair on, two things are tried when a
    drone finds no free slot:
    - a swap: the drones placed before it are tried, the last placed first. Without that
      drone the new one may find a free slot, and the drone taken out a free slot again
      among all the others. The first swap that frees both is kept.
    - when a drone still has no free slot, it goes to the front of the order (placed
      before every other, at its latest free slot) and the pass is made again. The drone
      that fails next joins it, until no drone fails or one fails a second time. The pass
      with the fewest drones without a free slot is kept.
    """
    booked = list(booked_starts)
    order = sorted(deadlines, key=lambda d: (-d[1], d[0]))
    placed, failed = _greedy(now, order, booked, length, capacity, tolerance, swap=repair)

    if failed and repair:
        best = (placed, failed)
        first = []
        for _ in range(len(order)):
            if failed[0] in first:
                break
            first.append(failed[0])
            front = sorted((d for d in order if d[0] in first), key=lambda d: first.index(d[0]))
            placed, failed = _greedy(now, front + [d for d in order if d[0] not in first], booked,
                                     length, capacity, tolerance, swap=True)
            if len(failed) < len(best[1]):
                best = (placed, failed)
            if not failed:
                break
        placed = best[0]

    return {drone_id: ChargingSlot(start=start, leave=start - flight_time, latest_arrival=latest_arrival)
            for drone_id, start, _, latest_arrival, flight_time in placed}


class CentralizedWorld:
    """
    The global map, the registry of the drones and the charging scheduler of one
    simulation, see the module docstring. The constants are read from the first drone
    that registers: every drone of a simulation has the same configuration.
    """

    ##### Seconds added to every planned charge, see plan_charging #####
    SLOT_GUARD = 15.0
    APPROACH_EXTRA_TIME = 0.0
    ##### The deadlines are planned with a drain this much above DISCHARGE_RATE. DISCHARGE_RATE #####
    ##### is right on average (measured 2026-11-15, 6 runs of 3000 s: 0.999 of it), but over    #####
    ##### 100 s of MAPPING the drain reaches 1.08 of it at the 95th percentile, 1.12 at the     #####
    ##### 99th and 1.22 at most. With the nominal rate a deadline moves earlier whenever the     #####
    ##### drone drains faster, after the drones planned around it may already have left, and   #####
    ##### the slot it had can no longer be reached (seen: 5 drones charging at Nc = 3). Planned #####
    ##### with a drain above the real one, a deadline only moves later, and a slot that was     #####
    ##### reachable stays reachable. It costs (factor - 1) * DISCHARGE_RATE * flight time of    #####
    ##### battery at the arrival, about 0.01.                                                   #####
    PLANNING_DRAIN_FACTOR = 1.15

    def __init__(self):
        ##### The base node, whose pads and waiting room the scheduler reads. main.py sets it once the  #####
        ##### simulation is built, before it starts, see recharge_base.py                              #####
        self.base = None
        self.drones = {}
        self.map = None
        self.is_cell_visited = None
        self._last_growth = None
        ##### Every slot handed to a drone when it left, for the report and the tests #####
        self.departures = []
        ##### The slot start of every drone GOING_TO_BASE, set when it leaves, see booked_starts #####
        self.planned_arrival = {}

    ##### Registry #####
    def register(self, drone) -> None:
        """Called by every drone in initialize. The first one also builds the global map."""
        if self.map is None:
            self.map = np.zeros((drone.MAP_WIDTH, drone.MAP_HEIGHT, 2))
            self.map[:, :, 0] = drone._config["initial_uncertainty"]
            self.is_cell_visited = np.zeros((drone.MAP_WIDTH, drone.MAP_HEIGHT))
            self.uncertainty_rate = drone.UNCERTAINTY_RATE
            self.growth_period = drone.VANISHING_UPDATE_TIME
            self.base_position = tuple(drone.CHARGING_BASE_POSITION)
            self.capacity = drone.Nc_CHARGING_BASE
            self.time_to_recharge = drone.TIME_TO_RECHARGE
            self.discharge_rate = drone.DISCHARGE_RATE
            self.reserve = drone.MIN_CHARGE_MARGIN
            self.speed = drone.speed_command
            self.braking_time = drone.speed_command / (2 * drone.BASE_APPROACH_DECELERATION)
            self.status_values = {status.name: status for status in type(drone.status)}
        self.drones[drone.provider.get_id()] = drone

    def others(self, drone_id: int) -> list:
        return [self.drones[i] for i in sorted(self.drones) if i != drone_id]

    ##### Global map #####
    def grow_uncertainty(self, now: float) -> None:
        """
        Every drone calls it on its own vanishing timer, all of them at the same instants.
        Only the first call of each period grows the map, so it grows by UNCERTAINTY_RATE
        once per period, as each drone's own map does in swarm.
        """
        if self._last_growth is not None and now - self._last_growth < 0.5 * self.growth_period:
            return
        self._last_growth = now
        self.map[:, :, 0] = self.map[:, :, 0] + self.uncertainty_rate
        self.is_cell_visited[self.map[:, :, 1] > 0.0] = 1

    ##### True state of the drones #####
    def flight_time_to_base(self, drone) -> float:
        """
        Expected flight time from the drone to the base: the straight line at cruise speed
        plus the time lost braking on the approach (the formula of EXPECTED_TIME_TO_BASE),
        plus APPROACH_EXTRA_TIME. Before its first telemetry the drone is on the base, where
        it takes off from.
        """
        if drone.drone_position is None:
            distance = 0.0
        else:
            distance = float(np.hypot(drone.drone_position[0] - self.base_position[0],
                                      drone.drone_position[1] - self.base_position[1]))
        return distance / self.speed + self.braking_time + self.APPROACH_EXTRA_TIME

    def battery_table(self, now: float) -> np.ndarray:
        """
        The table swarm_battery_status of swarm, (N, 3), filled with the truth: every row is
        the real battery of the drone, observed now, with its real status.
        """
        table = np.zeros((len(self.drones), 3))
        for i, drone in self.drones.items():
            table[i] = (drone.battery.battery_status, now, drone.status.value)
        return table

    ##### Charging scheduler #####
    def booked_starts(self, now: float) -> list:
        """
        Start of the charges already decided: drones CHARGING, when they arrived, and drones
        GOING_TO_BASE, at the slot they left for. A drone on its way keeps the slot it was
        given rather than a new prediction every tick: while it turns towards the base its
        distance barely drops, and now + flight_time_to_base would slide later by up to a
        second per second, moving the slots of the others with it (seen: a 2 s slide on a
        base full to capacity pushed another drone 105 s earlier). Only a drone really
        behind its slot moves its booking later, to now + the straight flight left.
        """
        charging = self.status_values["CHARGING"]
        going = self.status_values["GOING_TO_BASE"]
        starts = []
        for drone_id, drone in self.drones.items():
            if drone.status == charging:
                starts.append(drone.charging_start_time)
            elif drone.status == going:
                straight = now + self.flight_time_to_base(drone) - self.APPROACH_EXTRA_TIME
                starts.append(max(self.planned_arrival.get(drone_id, straight), straight))
        return starts + self.waiting_room_starts(now)

    def waiting_room_starts(self, now: float) -> list:
        """
        When the drones of the waiting room will start charging (since 2026-11-24, see
        recharge_base.py). They take the pads in the order they landed, each one as soon as
        a pad frees: at the end of a charge on the base, or of a charge of the waiting room
        before it. The start is never before now.
        """
        charging_starts = [start for start, _ in self.base.charging.values()]
        return waiting_room_starts(charging_starts, len(self.base.waiting), self.base.pads,
                                   self.time_to_recharge, now)

    def plan(self, now: float) -> dict:
        """The ChargingSlot of every MAPPING drone, from the true state at now, see plan_charging."""
        mapping = self.status_values["MAPPING"]
        mapping_drones = [(i, drone.battery.battery_status, self.flight_time_to_base(drone))
                          for i, drone in sorted(self.drones.items()) if drone.status == mapping]
        return plan_charging(now, mapping_drones, self.booked_starts(now), self.capacity,
                             self.time_to_recharge, self.SLOT_GUARD,
                             self.PLANNING_DRAIN_FACTOR * self.discharge_rate, self.reserve)

    def time_to_leave(self, drone_id: int, now: float) -> bool:
        """
        Whether the MAPPING drone drone_id has to leave for the base now. Asked by the drone
        on every mobility tick, so it leaves at most one tick after its planned time.
        """
        slot = self.plan(now).get(drone_id)
        if slot is None or slot.leave > now:
            return False
        ##### It arrives at the start of its slot, or as soon as it can when it is late for it #####
        self.planned_arrival[drone_id] = max(slot.start, now + self.flight_time_to_base(self.drones[drone_id]))
        self.departures.append({"drone": drone_id, "t": now, "slot_start": slot.start,
                                "latest_arrival": slot.latest_arrival,
                                "flight_time": self.flight_time_to_base(self.drones[drone_id]),
                                "battery": float(self.drones[drone_id].battery.battery_status)})
        return True
