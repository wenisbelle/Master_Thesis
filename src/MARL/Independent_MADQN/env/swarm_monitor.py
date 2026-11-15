"""
Swarm level quantities of a run, sampled while the simulation goes on.

SwarmMonitor is GlobalMapMonitor of Analytical/swarm/main.py (same samples, same
sums), so the cost of a run of the MARL system is the cost the GA of the
analytical system minimizes, and the two systems can be compared on it:

    cost = UNCERTAINTY_WEIGHT * integral of the global uncertainty
         + DEAD_DRONE_PENALTY * integral of the number of DEAD drones
         + integral of congestion_penalty_rate(drones CHARGING above Nc)

The cost weights are the ones of Analytical/swarm/main.py (CostWeights). They are
only used to report a run. The reward the agents learn from is computed by the
environment, see RewardWeights in mapping_environment.py.

It also accumulates the congestion of every sample until the environment reads
it at the end of a step (pop_step_congestion), for the reward.
"""

from dataclasses import dataclass
import numpy as np

from .gradysim_environment.protocol import DroneStatus


@dataclass(frozen=True)
class CostWeights:
    """The cost of Analytical/swarm/main.py, see simulation_cost_terms there."""
    uncertainty_weight: float = 0.00072          # UNCERTAINTY_WEIGHT, per unit of global uncertainty per second
    dead_drone_penalty: float = 1000.0           # DEAD_DRONE_PENALTY, per dead drone per second
    charging_congestion_penalty: float = 30.0    # CHARGING_CONGESTION_PENALTY, K of congestion_penalty_rate
    charging_congestion_steepness: float = 5.0   # CHARGING_CONGESTION_STEEPNESS, beta of congestion_penalty_rate


COST_TERMS = ("uncertainty", "dead", "charging")


def congestion_penalty_rate(excess: int, number_of_drones: int, penalty: float, steepness: float) -> float:
    """
    Cost per simulated second of having excess drones CHARGING above the base
    capacity: K * (exp(beta * excess / number_of_drones) - 1), as in
    Analytical/swarm/main.py. Zero with no excess, the same excess costs more in a
    smaller swarm.
    """
    return penalty * (np.exp(steepness * excess / number_of_drones) - 1.0)


class SwarmMonitor:
    """
    GlobalMapMonitor of Analytical/swarm/main.py, see the module docstring.

    The global map is the element-wise minimum of the uncertainty maps of every
    drone, DEAD ones included, as there.
    """

    def __init__(self, simulation, number_of_drones: int, sample_interval: float,
                 nc_charging_base: int, cost_weights: CostWeights = CostWeights(),
                 reward_congestion: tuple = (0.0, 5.0)):
        self.sim = simulation
        self.number_of_drones = number_of_drones
        self.sample_interval = sample_interval
        self.nc_charging_base = nc_charging_base
        self.cost_weights = cost_weights
        ##### (K, beta) of the congestion term of the reward, see RewardWeights #####
        self.reward_congestion = reward_congestion
        self._protocols = None

        self.times = []
        self.uncertainty = []
        self.unvisited = []
        self.accomulated_uncertainty = 0.0
        self.map = None
        self.visited = None

        self.dead_drones = []
        self.charging_drones = []
        self.accomulated_dead_time = 0.0
        self.accomulated_charging_excess = 0.0
        self.accomulated_charging_penalty = 0.0

        ##### Reward of the samples since the last pop_step_congestion #####
        self._step_congestion = 0.0
        self._step_congestion_share = np.zeros(number_of_drones)

    def protocols(self) -> list:
        if self._protocols is None:
            ##### The protocols only exist once the simulation has been initialized #####
            self._protocols = [self.sim.get_node(i).protocol_encapsulator.protocol
                               for i in range(self.number_of_drones)]
        return self._protocols

    def global_map(self) -> np.ndarray:
        return np.min([p.map[:, :, 0] for p in self.protocols()], axis=0)

    def sample(self, time: float) -> None:
        protocols = self.protocols()
        self.map = self.global_map()
        self.visited = np.any([p.is_cell_visited > 0 for p in protocols], axis=0)

        uncertainty = float(self.map.sum())
        self.accomulated_uncertainty += uncertainty * self.sample_interval

        self.times.append(float(time))
        self.uncertainty.append(uncertainty)
        self.unvisited.append(int(self.visited.size - self.visited.sum()))

        ##### Penalties. A dead drone counts for every sample until the end of #####
        ##### the run, so the earlier it dies the more it costs.                #####
        dead = sum(p.status == DroneStatus.DEAD for p in protocols)
        charging_mask = np.array([p.status == DroneStatus.CHARGING for p in protocols])
        charging = int(charging_mask.sum())
        self.accomulated_dead_time += dead * self.sample_interval
        excess = max(0, charging - self.nc_charging_base)
        self.accomulated_charging_excess += excess * self.sample_interval
        self.accomulated_charging_penalty += (
            congestion_penalty_rate(excess, self.number_of_drones,
                                    self.cost_weights.charging_congestion_penalty,
                                    self.cost_weights.charging_congestion_steepness)
            * self.sample_interval)

        self.dead_drones.append(dead)
        self.charging_drones.append(charging)

        ##### Congestion of the reward: the swarm pays it, and it is shared by the drones charging #####
        if excess > 0:
            penalty, steepness = self.reward_congestion
            congestion = congestion_penalty_rate(excess, self.number_of_drones, penalty, steepness) * self.sample_interval
            self._step_congestion += congestion
            self._step_congestion_share[charging_mask] += congestion / charging

    def pop_step_congestion(self) -> tuple[float, np.ndarray]:
        """Congestion of the reward since the last call: of the swarm, and the share of every drone."""
        congestion, share = self._step_congestion, self._step_congestion_share
        self._step_congestion = 0.0
        self._step_congestion_share = np.zeros(self.number_of_drones)
        return congestion, share

    @property
    def final_uncertainty(self) -> float:
        return self.uncertainty[-1] if self.uncertainty else float("nan")

    @property
    def unvisited_cells(self) -> int:
        return self.unvisited[-1] if self.unvisited else 0

    def cost_terms(self) -> dict:
        """The weighted terms of the cost of the run, keyed by COST_TERMS, as simulation_cost_terms."""
        return dict(
            uncertainty=self.cost_weights.uncertainty_weight * self.accomulated_uncertainty,
            dead=self.cost_weights.dead_drone_penalty * self.accomulated_dead_time,
            charging=self.accomulated_charging_penalty,
        )

    def cost(self) -> float:
        return float(sum(self.cost_terms().values()))
