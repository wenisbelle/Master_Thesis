"""
Records the state of every drone during a simulation and writes it to a pickle
that the plot scripts in the plots directory read back.

The recorder pulls the state out of the running simulator, so it needs the
simulation object and a driver that calls snapshot() every few simulated
seconds (see Analytical/main_test.py). Besides the uncertainty maps and the
positions, it samples the per-drone scalars listed in SERIES, which is what
plots/plot_drones.py and plots/plot_more_metrics.py plot.

Next to the map of each drone it also stores the global map of the swarm, the
element-wise minimum of all of them: the cell as the best informed drone knows
it. No drone ever holds that map, it only exists as a property of the system.

Two more things are recorded for the energy analysis, read by
plots/plot_estimation.py and plots/plot_returns.py:
- What each drone believes about the battery and state of the others (BELIEFS)
  and the swarm terms it decides with, estimated and true (SWARM_TERMS). The
  recorder sees the true state of every drone, no drone does, so the error of
  the estimate can be measured. A protocol without an estimator (egocentric)
  records NaN there.
- The moments a drone chooses the base, reaches it, is sent back to mapping on
  the way, or dies (record_return_event), with the battery it had then. They
  are caught by the driver when they happen, not on the snapshot grid.
Reading the state never changes it, so a recorded run is identical to an
unrecorded one with the same seed.
"""

import pickle
import numpy as np


def _read(reader, protocol):
    """A protocol that does not expose one of the series is recorded as NaN."""
    try:
        return float(reader(protocol))
    except (AttributeError, TypeError, ValueError):
        return float("nan")


class SimulationRecorder:
    ##### Per-drone scalars sampled on every snapshot. The plots look them up #####
    ##### by these names, so a new variable only has to be added here to      #####
    ##### become plottable.                                                   #####
    SERIES = {
        "battery": lambda p: p.battery.battery_status,
        "status": lambda p: p.status.value,
        "uncertainty": lambda p: p.map[:, :, 0].sum(),
        "accomulated_uncertainty": lambda p: p.accomulated_uncertainty,
        "visited_cells": lambda p: p.is_cell_visited.sum(),
        "distance": lambda p: p.total_distance_traveled,
        ##### Effective margin of recharge_base_fitness: the reserve the drone would #####
        ##### keep if it decided now, from its own estimate of the swarm. The drone  #####
        ##### only uses it while MAPPING, the plots leave the other states out.     #####
        "charge_margin": lambda p: p.fitness.charge_margin(
            p.battery.battery_status,
            *p.aproximate_battery_consumption_model(p.swarm_battery_status,
                                                    p.provider.current_time())),
    }

    ##### Constants of the run, drawn as reference lines. Read once from the #####
    ##### protocol that ran, so they match it even if the defaults change.   #####
    PARAMETERS = {
        "charge_margin": lambda p: p.fitness.CHARGE_MARGIN,
        "min_charge_margin": lambda p: p.fitness.MIN_CHARGE_MARGIN,
        "Nc_charging_base": lambda p: p.fitness.Nc_CHARGING_BASE,
        ##### The constants of the battery model the drones decide with #####
        "discharge_rate": lambda p: p.DISCHARGE_RATE,
        "drone_speed": lambda p: p.speed_command,
        "time_to_recharge": lambda p: p.TIME_TO_RECHARGE,
        "expected_time_to_base": lambda p: p.EXPECTED_TIME_TO_BASE,
        "dead_battery_threshold": lambda p: p.DEAD_BATTERY_THRESHOLD,
    }

    ##### What drone i believes about every drone j: its swarm_battery_status row for j, #####
    ##### projected to the current time by estimate_drone_battery, the way the drone      #####
    ##### itself does when it decides. Stored as (T, N) per drone i, one column per j,     #####
    ##### NaN on j == i (a drone never estimates itself, it reads its own battery).        #####
    #####   belief_battery   estimated battery of j                                        #####
    #####   belief_charging  1 when j is estimated to be on the base, charging or waiting   #####
    #####   belief_age       time since the row was written. For a CHARGING row that is    #####
    #####                    the time since j's charge started, the row keeps that time.   #####
    #####                    Negative for a WAITING row: the time left until it starts     #####
    #####   belief_status    status of j in the row (DroneStatus value), DEAD included      #####
    BELIEFS = ("belief_battery", "belief_charging", "belief_age", "belief_status")

    ##### Inputs of the swarm and congestion terms of charge_margin, (T,) per drone: the  #####
    ##### mean battery of the other drones alive and how many of them are on the base   #####
    ##### (charging or waiting, since 2026-11-24), as                                    #####
    ##### the drone estimates them (what it decides with) and as they really are, and   #####
    ##### the margin each pair gives. The true values are computed like               #####
    ##### aproximate_battery_consumption_model does, from the real state of the drones. #####
    SWARM_TERMS = ("mean_battery_estimate", "mean_battery_true",
                   "drones_charging_estimate", "drones_charging_true",
                   "charge_margin_estimate", "charge_margin_true")

    def __init__(self, simulation, num_drones, map_width, map_height, distance_between_cells,
                 status_labels=None):
        self.sim = simulation
        self.N, self.W, self.H, self.d = num_drones, map_width, map_height, distance_between_cells
        ##### {status value: name}, so the plots can label the states without #####
        ##### importing the protocol.                                          #####
        self.status_labels = dict(status_labels or {})
        self.times = []
        self.maps      = {i: [] for i in range(num_drones)}   # own uncertainty map
        self.true_pos  = {i: [] for i in range(num_drones)}   # own (cx, cy) in cell coords
        self.world_pos = {i: [] for i in range(num_drones)}   # own (x, y, z) in metres
        self.series    = {i: {name: [] for name in self.SERIES} for i in range(num_drones)}
        self.beliefs   = {i: {name: [] for name in self.BELIEFS} for i in range(num_drones)}
        self.swarm_terms = {i: {name: [] for name in self.SWARM_TERMS} for i in range(num_drones)}
        self.decisions = {i: [] for i in range(num_drones)}
        self.returns   = {i: [] for i in range(num_drones)}
        ##### Status values the true swarm terms need. NaN never matches, so without #####
        ##### the labels no drone counts as dead or charging.                        #####
        status_values = {name: value for value, name in self.status_labels.items()}
        self._dead = status_values.get("DEAD", np.nan)
        ##### On the base: charging, or in the waiting room (since 2026-11-24) #####
        self._on_base = [status_values.get("CHARGING", np.nan), status_values.get("WAITING", np.nan)]
        ##### Swarm level, one entry per snapshot, aligned with self.times #####
        self.global_maps = []                                 # min over the drones, per cell
        self.global_uncertainty = []
        self.unvisited = []

    def _to_cell(self, wx, wy):
        return (wx + self.W * self.d / 2) / self.d, (wy + self.H * self.d / 2) / self.d

    def _protocols(self):
        return [self.sim.get_node(i).protocol_encapsulator.protocol for i in range(self.N)]

    @staticmethod
    def _true_state(protocols):
        """The real battery and status value of every drone, what no drone knows."""
        battery = np.array([_read(lambda p: p.battery.battery_status, p) for p in protocols])
        status = np.array([_read(lambda p: p.status.value, p) for p in protocols])
        return battery, status

    def _beliefs(self, drone_id, protocol):
        """The BELIEFS of drone_id about every drone, see BELIEFS."""
        beliefs = {name: np.full(self.N, np.nan) for name in self.BELIEFS}
        estimator = getattr(protocol, "estimate_drone_battery", None)
        rows = getattr(protocol, "swarm_battery_status", None)
        if estimator is None or rows is None:
            return beliefs

        now = protocol.provider.current_time()
        for j in range(self.N):
            if j == drone_id:
                continue
            battery, observation_time, status = rows[j]
            estimate, charging = estimator(battery, observation_time, status, now)
            beliefs["belief_battery"][j] = estimate
            beliefs["belief_charging"][j] = float(charging)
            beliefs["belief_age"][j] = now - observation_time
            beliefs["belief_status"][j] = status
        return beliefs

    def _swarm_terms(self, drone_id, protocols, battery, status):
        """
        The SWARM_TERMS of drone_id. The estimate comes from its own
        aproximate_battery_consumption_model. The truth is computed the same way
        from the real state: the other drones that are not DEAD, the drone's own
        battery and nobody charging when there is none.
        """
        protocol = protocols[drone_id]
        try:
            mean_estimate, charging_estimate = protocol.aproximate_battery_consumption_model(
                protocol.swarm_battery_status, protocol.provider.current_time())
        except (AttributeError, TypeError, ValueError):
            mean_estimate, charging_estimate = np.nan, np.nan

        alive_others = (np.arange(self.N) != drone_id) & (status != self._dead)
        if alive_others.any():
            mean_true = float(np.mean(battery[alive_others]))
            charging_true = int(np.sum(np.isin(status[alive_others], self._on_base)))
        else:
            mean_true, charging_true = float(battery[drone_id]), 0

        own_battery = battery[drone_id]
        return {
            "mean_battery_estimate": float(mean_estimate),
            "mean_battery_true": mean_true,
            "drones_charging_estimate": float(charging_estimate),
            "drones_charging_true": float(charging_true),
            "charge_margin_estimate": _read(
                lambda p: p.fitness.charge_margin(own_battery, mean_estimate, charging_estimate), protocol),
            "charge_margin_true": _read(
                lambda p: p.fitness.charge_margin(own_battery, mean_true, charging_true), protocol),
        }

    def snapshot(self, time=None):
        ##### The driver samples on a fixed grid, so it passes the nominal time #####
        ##### of the sample. Without it the timestamp of the last event is used. #####
        t = self.sim._current_timestamp if time is None else float(time)
        self.times.append(t)
        unc_maps, visited = [], []
        protocols = self._protocols()
        battery, status = self._true_state(protocols)
        for i, p in enumerate(protocols):
            for name, values in self._beliefs(i, p).items():
                self.beliefs[i][name].append(values)
            for name, value in self._swarm_terms(i, protocols, battery, status).items():
                self.swarm_terms[i][name].append(value)

            umap = p.map[:, :, 0].astype(np.float32)
            self.maps[i].append(umap.copy())
            unc_maps.append(umap)
            visited.append(p.is_cell_visited > 0)
            if p.drone_position is not None:
                self.true_pos[i].append(self._to_cell(p.drone_position[0], p.drone_position[1]))
                self.world_pos[i].append(tuple(float(c) for c in p.drone_position[:3]))
            else:
                self.true_pos[i].append((np.nan, np.nan))
                self.world_pos[i].append((np.nan, np.nan, np.nan))

            for name, reader in self.SERIES.items():
                self.series[i][name].append(_read(reader, p))

        # global reductions, aligned with self.times
        global_map = np.min(unc_maps, axis=0)             # best-known uncertainty per cell
        self.global_maps.append(global_map.astype(np.float32))
        self.global_uncertainty.append(float(global_map.sum()))
        any_visited = np.any(visited, axis=0)             # visited by ≥1 drone
        self.unvisited.append(int(any_visited.size - any_visited.sum()))

    def record_decision(self, drone_id, time, current_position, target_position):
        """
        Stores one change of destination, in cell coordinates, the way
        plots/plot_decisions.py expects it. Both positions are in metres.
        """
        self.decisions[drone_id].append({
            "t": float(time),
            "current_cell": self._to_cell(float(current_position[0]), float(current_position[1])),
            "target_cell": self._to_cell(float(target_position[0]), float(target_position[1])),
        })

    def record_return_event(self, drone_id, event, time, trigger, sender=None, previous_status=None):
        """
        Stores one change of the drone's relation to the base, at the moment the
        driver sees it:
            "decision"  it chose the base (status became GOING_TO_BASE)
            "arrival"   it reached the base and starts charging
            "abort"     it was sent back to mapping on the way to the base
            "death"     it ran out of battery, previous_status says doing what
        trigger is what the drone was handling when it changed ("own" for its own
        choice on the mobility timer, "encounter" when it decided for both drones of
        an encounter, "sent" when the other drone of the encounter decided, with its
        id in sender, "charging_notification", "after_recharge"). Stored with the
        battery it had, the distance to the base and the energy the drone expects
        to spend to reach it, and its swarm terms, estimated and true. For a "sent"
        decision those are the receiver's: the sender decided with its own.
        """
        protocols = self._protocols()
        protocol = protocols[drone_id]
        battery, status = self._true_state(protocols)

        position = protocol.drone_position
        base = getattr(protocol, "CHARGING_BASE_POSITION", None)
        if position is None or base is None:
            distance = float("nan")
        else:
            distance = float(np.hypot(position[0] - base[0], position[1] - base[1]))

        self.returns[drone_id].append({
            "t": float(time),
            "event": event,
            "trigger": trigger,
            "sender": sender,
            "previous_status": previous_status,
            "battery": float(battery[drone_id]),
            "distance_to_base": distance,
            "energy_to_base": _read(lambda p: distance / p.speed_command * p.DISCHARGE_RATE, protocol),
            **self._swarm_terms(drone_id, protocols, battery, status),
        })

    def drone_arrays(self):
        """
        Every per-drone series except the maps, as arrays: the "states" of the
        saved recording minus maps and positions. Also what main_test reports from.
        """
        arrays = {}
        for i in range(self.N):
            arrays[i] = {name: np.array(values, dtype=np.float32)
                         for name, values in {**self.series[i], **self.swarm_terms[i]}.items()}
            for name, values in self.beliefs[i].items():
                arrays[i][name] = (np.stack(values) if values
                                   else np.empty((0, self.N))).astype(np.float32)
        return arrays

    def save(self, path, results_aggregator=None):
        ##### Decisions recorded here win, the aggregator is only read for  #####
        ##### protocols that fill it in themselves.                          #####
        decisions = {i: list(self.decisions[i]) for i in range(self.N) if self.decisions[i]}
        for drone_id, recorded in (results_aggregator or {}).get("decisions", {}).items():
            decisions.setdefault(drone_id, []).extend(recorded)

        ##### Every drone runs with the same configuration, the first one is enough #####
        first_protocol = self.sim.get_node(0).protocol_encapsulator.protocol
        parameters = {name: _read(reader, first_protocol) for name, reader in self.PARAMETERS.items()}
        drone_arrays = self.drone_arrays()

        data = {
            "meta": {"num_drones": self.N, "map_width": self.W, "map_height": self.H,
                     "distance_between_cells": self.d,
                     "times": np.array(self.times),
                     ##### (T, W, H), the swarm map the plots draw next to the #####
                     ##### individual ones                                     #####
                     "global_maps": np.stack(self.global_maps),
                     "global_uncertainty": np.array(self.global_uncertainty),
                     "unvisited": np.array(self.unvisited),
                     "status_labels": dict(self.status_labels),
                     "parameters": parameters},
            "states": {i: {
                "maps":          np.stack(self.maps[i]),
                "true_pos_cell": np.array(self.true_pos[i], dtype=np.float32),
                "position":      np.array(self.world_pos[i], dtype=np.float32),
                **drone_arrays[i],
            } for i in range(self.N)},
            "decisions": decisions,
            "returns": {i: list(self.returns[i]) for i in range(self.N)},
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
