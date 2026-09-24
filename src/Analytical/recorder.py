"""
Records the state of every drone during a simulation and writes it to a pickle
that the plot scripts in the logs directory read back.

The recorder pulls the state out of the running simulator, so it needs the
simulation object and a driver that calls snapshot() every few simulated
seconds (see Analytical/main_test.py). Besides the uncertainty maps and the
positions, it samples the per-drone scalars listed in SERIES, which is what
logs/plot_drones.py plots.

Next to the map of each drone it also stores the global map of the swarm, the
element-wise minimum of all of them: the cell as the best informed drone knows
it. No drone ever holds that map, it only exists as a property of the system.
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
    }

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
        self.decisions = {i: [] for i in range(num_drones)}
        ##### Swarm level, one entry per snapshot, aligned with self.times #####
        self.global_maps = []                                 # min over the drones, per cell
        self.global_uncertainty = []
        self.unvisited = []

    def _to_cell(self, wx, wy):
        return (wx + self.W * self.d / 2) / self.d, (wy + self.H * self.d / 2) / self.d

    def snapshot(self, time=None):
        ##### The driver samples on a fixed grid, so it passes the nominal time #####
        ##### of the sample. Without it the timestamp of the last event is used. #####
        t = self.sim._current_timestamp if time is None else float(time)
        self.times.append(t)
        unc_maps, visited = [], []
        for i in range(self.N):
            p = self.sim.get_node(i).protocol_encapsulator.protocol
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
        logs/plot_decisions.py expects it. Both positions are in metres.
        """
        self.decisions[drone_id].append({
            "t": float(time),
            "current_cell": self._to_cell(float(current_position[0]), float(current_position[1])),
            "target_cell": self._to_cell(float(target_position[0]), float(target_position[1])),
        })

    def save(self, path, results_aggregator=None):
        ##### Decisions recorded here win, the aggregator is only read for  #####
        ##### protocols that fill it in themselves.                          #####
        decisions = {i: list(self.decisions[i]) for i in range(self.N) if self.decisions[i]}
        for drone_id, recorded in (results_aggregator or {}).get("decisions", {}).items():
            decisions.setdefault(drone_id, []).extend(recorded)

        data = {
            "meta": {"num_drones": self.N, "map_width": self.W, "map_height": self.H,
                     "distance_between_cells": self.d,
                     "times": np.array(self.times),
                     ##### (T, W, H), the swarm map the plots draw next to the #####
                     ##### individual ones                                     #####
                     "global_maps": np.stack(self.global_maps),
                     "global_uncertainty": np.array(self.global_uncertainty),
                     "unvisited": np.array(self.unvisited),
                     "status_labels": dict(self.status_labels)},
            "states": {i: {
                "maps":          np.stack(self.maps[i]),
                "true_pos_cell": np.array(self.true_pos[i], dtype=np.float32),
                "position":      np.array(self.world_pos[i], dtype=np.float32),
                **{name: np.array(values, dtype=np.float32)
                   for name, values in self.series[i].items()},
            } for i in range(self.N)},
            "decisions": decisions,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
