"""
TorchRL environment of the mapping swarm with energy, the scenario of Analytical/swarm.

The drones run the protocol of Analytical/swarm (gradysim_environment/protocol.py):
same map, battery, charging base, gossip, battery estimation and encounters. The
only difference is who chooses the destinations: the policy, through this
environment, instead of the fitness function.

How a step works
----------------
A drone that needs a destination appends a DecisionRequest to decision_requests
(protocol.request_decision). The simulation stops right after that event
(base_gradys_env.step_simulation) and the step returns with the drone flagged
(observation "decision_flag"), its observation built by the drone that decides
(protocol.observe). The next step applies the action of every flagged agent
(protocol.apply_decision) and goes on. A step also ends every
algorithm_iteration_interval simulated seconds, so the rewards keep flowing.

At an encounter the drone with the highest id decides for both, as in the
analytical protocol: its own destination first (ENCOUNTER), then, in the next
step and at the same simulated time, the one of the other drone (ENCOUNTER_PARTNER,
the other drone's slot is flagged, with an observation built by the decider that
includes the destination it just chose).

decision_backend "analytical" replays the analytical protocol: the env takes the
action of the fitness function (protocol.analytical_action) and ignores the
actions given. With the same seed, base and parameters, the run is the run of
Analytical/swarm, bit for bit. It is how the protocol is checked
(check_equivalence.py) and a baseline for the tests.

Reward
------
Per agent and per step (RewardWeights):
    local  = uncertainty_scale * uncertainty removed from the agent's own map
             - its share of the congestion (it is CHARGING while more than Nc are)
             - death_penalty, on the step the agent dies (its last transition)
    global = uncertainty_scale * uncertainty removed from the global map / N
             - congestion of the swarm
             - death_penalty * drones that died in the step
The uncertainty removed is the decrease of the map plus the growth all cells had
in the step (UNCERTAINTY_RATE per cell every VANISHING_UPDATE_TIME): what the
cameras (and, for the own map, the gossip) took away. It is >= 0, so being alive
is never worse than being dead. The congestion is congestion_penalty_rate of the
drones CHARGING above Nc, per second, as the cost of the analytical system. The
orchestrator gives each agent local (Independent) or local + ratio * global (CTDE).

The cost of the analytical system is also computed on every run (SwarmMonitor),
so the two systems can be compared on the same scale.
"""

import dataclasses
import math
import random
from typing import Callable, Optional

import numpy as np
import torch
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import (
    DynamicVelocityMobilityConfiguration,
    DynamicVelocityMobilityHandler,
)
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from tensordict import TensorDictBase
from torchrl.data import Binary
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .environment import EndCause
from .metrics import make_data_collection_metrics_spec, EnvironmentMetricsSpec
from .swarm_monitor import SwarmMonitor, CostWeights
from .gradysim_environment.protocol import (
    Drone, DroneStatus, DecisionKind, DecisionRequest, drone_protocol_factory,
    MAP_CHANNELS, AGENT_STATE_DIM, ENCOUNTER_STATE_DIM, UNCERTAINTY_CLIP,
)
from .base_gradys_env import BaseGrADySEnvironment


@dataclasses.dataclass(slots=True)
class EpisodeAgentState():
    """Bookkeeping for a single stable agent slot in the current episode. Slot i is drone (node) i."""

    slot_index: int
    name: str
    exists: bool = False
    """Whether this agent slot is occupied by a drone in the current episode. If False, this slot is a
    placeholder (the episode has fewer drones than max_num_agents) and is never filled with meaningful data."""

    active: bool = False
    """Whether the drone is alive (not DEAD)."""

    node_id: int | None = None


@dataclasses.dataclass(frozen=True)
class RewardWeights:
    """Weights of the reward, see the module docstring."""
    uncertainty_scale: float = 0.01           # per unit of uncertainty removed
    death_penalty: float = 20.0               # once, when a drone dies
    charging_congestion_penalty: float = 0.1  # K of congestion_penalty_rate, per second
    charging_congestion_steepness: float = 5.0  # beta of congestion_penalty_rate


def sample_base_position(seed: int, map_width: int, map_height: int, distance_between_cells: float,
                         sigma: float) -> tuple:
    """
    sample_base_position of Analytical/swarm/main.py: a Gaussian centred in the middle
    of the map, snapped to the nearest cell centre, drawn again when it falls outside
    the map. Only depends on the seed, so the same seed gives the same base as there.
    """
    rng = np.random.default_rng(seed)
    cells = np.array([map_width, map_height])
    map_center_offset = cells * distance_between_cells / 2
    while True:
        cell = np.round((rng.normal(0.0, sigma, size=2) + map_center_offset) / distance_between_cells)
        if np.all(cell >= 0) and np.all(cell < cells):
            return tuple(float(c) for c in cell * distance_between_cells - map_center_offset)


@dataclasses.dataclass
class MappingEnvironmentConfig:
    """
    Scenario of the environment. The defaults are the scenario of Analytical/swarm/main.py,
    see config.py, which train.py and main_test.py fill this from.
    """

    render_mode: Optional[str] = None  # "visual": gradysim visualization, in real time
    algorithm_iteration_interval: float = 1.0  # longest step, simulated seconds
    sample_interval: float = 1.0  # simulated seconds between two samples of the swarm, GLOBAL_MAP_SAMPLE_INTERVAL

    ##### Swarm. The number of drones of an episode is drawn from num_agents_choices, the     #####
    ##### tensors have max(num_agents_choices) slots.                                          #####
    num_agents_choices: tuple = (3, 5, 7, 10)
    map_width: int = 100
    map_height: int = 100
    distance_between_cells: float = 20
    large_map_size: int = 50   # side of the whole map the network sees, averaged in blocks
    action_map_size: int = 21  # side of the window of cells a drone chooses from, odd
    uncertainty_rate: float = 0.001
    vanishing_update_time: float = 1.0
    initial_uncertainty: float = 1.0
    communication_range: float = 100
    drone_speed: float = 15.0
    drone_max_acceleration: float = 3.0
    discharge_rate: float = 0.00073
    min_charge_margin: float = 0.20
    charging_capacity: Optional[int] = None  # Nc, None: int(drones / 3)

    ##### Episode #####
    max_episode_length: float = 4000  # simulated seconds
    ##### seed fixes the scenario of every episode: wind, initial destinations, base and number of  #####
    ##### drones, and seeds random and numpy as Analytical/swarm/main.py does. None: a new scenario #####
    ##### every episode, drawn from the random stream of the process.                              #####
    seed: Optional[int] = None
    ##### None: the base is drawn every episode from its seed (sample_base_position) #####
    charging_base_position: Optional[tuple] = None
    base_position_sigma: float = 333.0
    ##### False keeps the simulation going until max_episode_length when every drone is dead, as #####
    ##### the analytical runs do (their cost keeps counting the dead drones). Tests set it False. #####
    end_when_all_dead: bool = True

    ##### Decisions #####
    decision_backend: str = "policy"  # "policy": the actions given, "analytical": the fitness function
    analytical_parameters: Optional[dict] = None  # the fitness parameters, for decision_backend "analytical"

    ##### Reward and cost #####
    reward: RewardWeights = dataclasses.field(default_factory=RewardWeights)
    cost: CostWeights = dataclasses.field(default_factory=CostWeights)

    ##### Tests: the protocol log ("test"), the live map plot, a wrapped protocol factory and #####
    ##### observers of the samples, see main_test.py                                         #####
    mode: str = "train"
    enable_map_plot: bool = False
    protocol_factory: Optional[Callable] = None
    observer_factory: Optional[Callable] = None

    @property
    def max_num_agents(self) -> int:
        return max(self.num_agents_choices)


class MappingEnvironment(BaseGrADySEnvironment, EnvBase):
    """
    See the module docstring. Per-episode agent state is in `episode_agents`.
    """

    _simulation_configuration: SimulationConfiguration

    batch_locked: bool = True

    def __init__(self, config: MappingEnvironmentConfig, *, device=None):
        BaseGrADySEnvironment.__init__(self, config.algorithm_iteration_interval,
                                       sample_interval=config.sample_interval,
                                       visual_mode=(config.render_mode == "visual"))
        EnvBase.__init__(self, device=device)

        if config.decision_backend not in ("policy", "analytical"):
            raise ValueError("decision_backend must be 'policy' or 'analytical'")
        if config.decision_backend == "analytical" and config.analytical_parameters is None:
            raise ValueError("decision_backend 'analytical' needs analytical_parameters")
        if config.map_width != config.map_height:
            raise ValueError("The protocol shares one centre offset between both axes, the map has to be square")

        self.config = config
        self.render_mode = config.render_mode
        self.max_num_agents = config.max_num_agents
        self.map_width = config.map_width
        self.map_height = config.map_height
        self.large_map_size = config.large_map_size
        self.action_map_size = config.action_map_size
        self.n_actions = config.action_map_size ** 2 + 1  # the cells of the window and the base
        self.distance_between_cells = config.distance_between_cells

        self.possible_agents = [f"drone{i}" for i in range(self.max_num_agents)]
        self.group_map = {"agents": self.possible_agents}

        self.episode_agents: list[EpisodeAgentState] = []
        self.monitor: Optional[SwarmMonitor] = None
        self.number_of_drones = 0
        self.charging_base_position = None
        self.episode_seed = None
        self.nc_charging_base = 0
        self.results_aggregator = {}
        ##### The request each flagged slot has to answer, {slot: DecisionRequest} #####
        self.flagged_requests: dict[int, DecisionRequest] = {}
        ##### Last action applied to every slot, and the decisions of the episode by kind #####
        self.last_applied_actions = {}
        self.decisions = {kind.name: 0 for kind in DecisionKind}

        self._metrics_spec: EnvironmentMetricsSpec = make_data_collection_metrics_spec()
        self._all_info_keys = self._metrics_spec.info_keys

        self._simulation_configuration = SimulationConfiguration(
            duration=config.max_episode_length,
            real_time=False,
            execution_logging=False,
        )

        self._build_specs()
        self._cached_reset_zero = self.full_observation_spec.zero()
        self._cached_reset_zero.update(self.full_done_spec.zero())

        self._cached_step_zero = self.full_observation_spec.zero()
        self._cached_step_zero.update(self.full_reward_spec.zero())
        self._cached_step_zero.update(self.full_done_spec.zero())

    def _existing_episode_agents(self) -> list[EpisodeAgentState]:
        """Return episode slots that exist in the current episode, in slot order."""
        return [agent for agent in self.episode_agents if agent.exists]

    def _active_episode_agents(self) -> list[EpisodeAgentState]:
        """Return the episode slots that are alive."""
        return [agent for agent in self.episode_agents if agent.exists and agent.active]

    def _agent_slot_tensor(self, agents: list[EpisodeAgentState]) -> torch.Tensor:
        """Return the given episode agents' slot indices on the environment device."""
        return torch.tensor([agent.slot_index for agent in agents], device=self.device, dtype=torch.long)

    def protocol(self, node_id: int):
        return self.simulator.get_node(node_id).protocol_encapsulator.protocol

    ##### Simulation #####
    def _build_simulation(self, builder: SimulationBuilder):
        """
        The simulation of create_and_run_simulation in Analytical/swarm/main.py: the same
        handlers, in the same order, and the drones taking off from the base.
        """
        config = self.config
        builder.add_handler(TimerHandler())
        # BatteryPowerPlugin requires DynamicVelocityMobilityHandler for velocity telemetry.
        mobility_config = DynamicVelocityMobilityConfiguration(
            update_rate=0.05,
            max_speed_xy=config.drone_speed,
            max_speed_z=10.0,
            max_acc_xy=config.drone_max_acceleration,
            max_acc_z=3.0,
            send_telemetry=True,
        )
        builder.add_handler(DynamicVelocityMobilityHandler(mobility_config))
        builder.add_handler(CommunicationHandler(CommunicationMedium(
            transmission_range=config.communication_range
        )))

        if self.render_mode == "visual":
            scenario_size = max(self.map_width / 2, self.map_height / 2) * self.distance_between_cells
            builder.add_handler(VisualizationHandler(VisualizationConfiguration(
                open_browser=False,
                x_range=(-scenario_size, scenario_size),
                y_range=(-scenario_size, scenario_size),
                z_range=(0, 100),
            )))

        self.results_aggregator = {}
        factory = config.protocol_factory or drone_protocol_factory
        ConfiguredDrone = factory(
            uncertainty_rate=config.uncertainty_rate,
            vanishing_update_time=config.vanishing_update_time,
            number_of_drones=self.number_of_drones,
            map_width=config.map_width,
            map_height=config.map_height,
            drone_speed=config.drone_speed,
            drone_max_acceleration=config.drone_max_acceleration,
            discharge_rate=config.discharge_rate,
            charging_base_position=self.charging_base_position,
            min_charge_margin=config.min_charge_margin,
            Nc_charging_base=self.nc_charging_base,
            results_aggregator=self.results_aggregator,
            decision_requests=self.decision_requests,
            action_map_size=config.action_map_size,
            large_map_size=config.large_map_size,
            analytical_parameters=config.analytical_parameters,
            mode=config.mode,
            enable_map_plot=config.enable_map_plot,
            initial_uncertainty=config.initial_uncertainty,
            seed=self.episode_seed,
        )

        ##### The drones take off from the recharge base. Node i is slot i. #####
        for agent in self._existing_episode_agents():
            agent.node_id = builder.add_node(
                ConfiguredDrone, (self.charging_base_position[0], self.charging_base_position[1], 0))

    def _make_observers(self) -> list:
        reward = self.config.reward
        self.monitor = SwarmMonitor(self.simulator, self.number_of_drones, self.config.sample_interval,
                                    self.nc_charging_base, cost_weights=self.config.cost,
                                    reward_congestion=(reward.charging_congestion_penalty,
                                                       reward.charging_congestion_steepness))
        observers = [self.monitor.sample]
        if self.config.observer_factory is not None:
            observers.append(self.config.observer_factory(self.simulator))
        return observers

    def _draw_episode(self) -> None:
        """Number of drones, base and seed of the new episode, see MappingEnvironmentConfig.seed."""
        config = self.config
        if config.seed is None:
            self.episode_seed = random.randrange(2**31)
            self.number_of_drones = random.choice(config.num_agents_choices)
        else:
            self.episode_seed = config.seed
            self.number_of_drones = random.Random(config.seed).choice(config.num_agents_choices)
            ##### As create_and_run_simulation, for anything else that draws from them #####
            random.seed(config.seed)
            np.random.seed(config.seed)
        if config.charging_base_position is not None:
            self.charging_base_position = tuple(config.charging_base_position)
        else:
            self.charging_base_position = sample_base_position(
                self.episode_seed, config.map_width, config.map_height, config.distance_between_cells,
                config.base_position_sigma)
        self.nc_charging_base = (config.charging_capacity if config.charging_capacity is not None
                                 else int(self.number_of_drones / 3))

    ##### Specs #####
    def _build_specs(self) -> None:
        device = self.device
        N = self.max_num_agents
        L = self.large_map_size
        A = self.action_map_size

        def bounded(shape, low, high):
            return Bounded(torch.full(shape, low, device=device), torch.full(shape, high, device=device),
                           shape, dtype=torch.float32, device=device)

        ##### Per agent, see protocol.observe #####
        obs_inner = {
            "large_map_patch": bounded((N, MAP_CHANNELS, L, L), -UNCERTAINTY_CLIP, UNCERTAINTY_CLIP),
            "small_map_patch": bounded((N, MAP_CHANNELS, A, A), -UNCERTAINTY_CLIP, UNCERTAINTY_CLIP),
            "agent_state": Unbounded(shape=(N, AGENT_STATE_DIM), device=device, dtype=torch.float32),
            "encounter_state": Unbounded(shape=(N, ENCOUNTER_STATE_DIM), device=device, dtype=torch.float32),
            ##### True when the agent has to decide in this step #####
            "decision_flag": Binary(n=1, shape=(N, 1), device=device, dtype=torch.bool),
            "valid_actions": Binary(n=self.n_actions, shape=(N, self.n_actions), device=device, dtype=torch.bool),
        }

        ##### For a centralized critic, not used by the DQN: the global map, every drone and the swarm #####
        obs_global = {
            "global_map": bounded((L, L), -UNCERTAINTY_CLIP, UNCERTAINTY_CLIP),
            ##### per slot: exists, alive, x, y in [0, 1], battery, going to base, charging #####
            "drones": Unbounded(shape=(N, 7), device=device, dtype=torch.float32),
            ##### mean global uncertainty, charging / N, Nc / N, dead / N, time / episode length #####
            "swarm": Unbounded(shape=(5,), device=device, dtype=torch.float32),
            "all_active": Categorical(n=2, shape=(N,), dtype=torch.bool, device=device),
        }

        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Composite(obs_inner, device=device),
                        "mask": Categorical(n=2, shape=(N,), dtype=torch.bool, device=device),
                        "info": Composite(
                            {key: Unbounded(shape=(N,), device=device, dtype=torch.float32)
                             for key in self._all_info_keys},
                            device=device,
                        ),
                    },
                    device=device,
                ),
                "global_state": Composite(obs_global, device=device),
            },
            device=device,
        )

        self.action_spec = Composite(
            {"agents": Composite(
                {"action": Categorical(n=self.n_actions, shape=(N, 1), dtype=torch.int64, device=device)},
                device=device)},
            device=device,
        )

        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {"reward": Unbounded(shape=(N, 1), device=device, dtype=torch.float32)},
                    device=device,
                ),
                "global_reward": Unbounded(shape=(1,), device=device, dtype=torch.float32),
            },
            device=device,
        )

        def flags(shape):
            return Categorical(n=2, shape=shape, dtype=torch.bool, device=device)

        self.done_spec = Composite(
            {
                "done": flags((1,)),
                "terminated": flags((1,)),
                "truncated": flags((1,)),
                "agents": Composite(
                    {"done": flags((N, 1)), "terminated": flags((N, 1)), "truncated": flags((N, 1))},
                    device=device,
                ),
            },
            device=device,
        )

    ##### Reset and step #####
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self._draw_episode()
        self.episode_agents = [
            EpisodeAgentState(slot_index=i, name=f"drone{i}",
                              exists=i < self.number_of_drones, active=i < self.number_of_drones)
            for i in range(self.max_num_agents)
        ]
        Drone.Number_of_Encounters = 0
        self.flagged_requests = {}
        self.last_applied_actions = {}
        self.decisions = {kind.name: 0 for kind in DecisionKind}

        self.reset_simulation(self._simulation_configuration)
        self._reset_reward_tracking()
        self._settle_analytical_decisions()
        self.flagged_requests = self._collect_flagged_requests()

        tensordict_out = self._cached_reset_zero.clone()
        self._fill_observation(tensordict_out, episode_ended=False)
        self._fill_done(tensordict_out, EndCause.NONE, [])
        self._fill_info(tensordict_out, EndCause.NONE, False)
        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._apply_actions(tensordict.get(("agents", "action")))
        status = self.step_simulation()
        status = self._settle_analytical_decisions(status)
        self.flagged_requests = self._collect_flagged_requests()

        local_rewards, global_reward, dying_agents = self._compute_rewards()

        end_cause = EndCause.NONE
        if not self._active_episode_agents() and self.config.end_when_all_dead:
            end_cause = EndCause.ALL_AGENTS_INACTIVE
        elif status.has_ended:
            end_cause = EndCause.TIMEOUT

        tensordict_out = self._cached_step_zero.clone()
        self._fill_observation(tensordict_out, episode_ended=end_cause != EndCause.NONE,
                               closing=dying_agents)
        self._fill_rewards(tensordict_out, local_rewards, global_reward)
        self._fill_done(tensordict_out, end_cause, dying_agents)
        self._fill_info(tensordict_out, end_cause, end_cause != EndCause.NONE)
        return tensordict_out

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    ##### Decisions #####
    def _apply_actions(self, actions: Optional[torch.Tensor]) -> None:
        """
        Applies the action of every pending request, see the module docstring. With the
        analytical backend the action is the one of the fitness function. Applying an
        ENCOUNTER request adds the ENCOUNTER_PARTNER request to decision_requests.
        """
        pending = list(self.decision_requests)
        self.decision_requests.clear()
        actions_cpu = None if actions is None else actions.detach().cpu()
        for request in pending:
            decider = self.protocol(request.decider_id)
            if self.config.decision_backend == "analytical":
                action = decider.analytical_action(request)
            else:
                action = int(actions_cpu[request.subject_id, 0].item())
            valid = decider.valid_actions(request.subject_position)
            if not valid[action]:
                raise ValueError(f"Drone {request.subject_id} was given the invalid action {action} "
                                 f"({request.kind.name}), the policy has to mask valid_actions")
            decider.apply_decision(request, action)
            self.last_applied_actions[request.subject_id] = action
            self.decisions[request.kind.name] += 1

    def _settle_analytical_decisions(self, status=None):
        """With the analytical backend the decisions are taken here, without returning to the caller."""
        if self.config.decision_backend != "analytical":
            return status
        while self.decision_requests:
            self._apply_actions(None)
            status = self.step_simulation()
        return status

    def _collect_flagged_requests(self) -> dict:
        flagged = {}
        for request in self.decision_requests:
            if request.subject_id in flagged:
                raise RuntimeError(f"Two decisions pending for drone {request.subject_id}")
            flagged[request.subject_id] = request
        return flagged

    def _own_request(self, agent: EpisodeAgentState) -> DecisionRequest:
        """A request of the drone for itself, for the observation that closes its transition at the end."""
        protocol = self.protocol(agent.node_id)
        position = protocol.drone_position if protocol.drone_position is not None else (*self.charging_base_position, 0.0)
        return DecisionRequest(kind=DecisionKind.OWN, decider_id=agent.node_id, subject_id=agent.node_id,
                               subject_position=np.array(position, dtype=float),
                               subject_battery=float(protocol.battery.battery_status),
                               subject_status=protocol.status.value,
                               time=self.current_time, trigger="end")

    ##### Rewards #####
    def _reset_reward_tracking(self) -> None:
        self._last_time = self.current_time
        self._last_global_uncertainty = float(self.monitor.global_map().sum())
        self._last_own_uncertainty = {agent.slot_index: float(self.protocol(agent.node_id).map[:, :, 0].sum())
                                      for agent in self._existing_episode_agents()}
        self.monitor.pop_step_congestion()
        self.episode_reward_terms = {"local_uncertainty": 0.0, "global_uncertainty": 0.0,
                                     "congestion": 0.0, "deaths": 0.0}
        self.episode_global_reward = 0.0

    def _compute_rewards(self):
        """
        The local reward of every slot alive at the start of the step, the global reward
        and the agents that died in the step. See the module docstring.
        """
        weights = self.config.reward
        time = self.current_time
        elapsed = time - self._last_time
        growth = (self.config.uncertainty_rate * self.map_width * self.map_height
                  * elapsed / self.config.vanishing_update_time)

        global_uncertainty = float(self.monitor.global_map().sum())
        removed_global = self._last_global_uncertainty - global_uncertainty + growth
        congestion, congestion_share = self.monitor.pop_step_congestion()

        local_rewards = {}
        dying_agents = []
        for agent in self._active_episode_agents():
            protocol = self.protocol(agent.node_id)
            own_uncertainty = float(protocol.map[:, :, 0].sum())
            removed_own = self._last_own_uncertainty[agent.slot_index] - own_uncertainty + growth
            self._last_own_uncertainty[agent.slot_index] = own_uncertainty
            reward = weights.uncertainty_scale * removed_own - congestion_share[agent.slot_index]
            if protocol.status == DroneStatus.DEAD:
                reward -= weights.death_penalty
                dying_agents.append(agent)
            local_rewards[agent.slot_index] = reward
            self.episode_reward_terms["local_uncertainty"] += weights.uncertainty_scale * removed_own

        for agent in dying_agents:
            agent.active = False

        global_reward = (weights.uncertainty_scale * removed_global / self.number_of_drones
                         - congestion - weights.death_penalty * len(dying_agents))

        self.episode_reward_terms["global_uncertainty"] += weights.uncertainty_scale * removed_global / self.number_of_drones
        self.episode_reward_terms["congestion"] += congestion
        self.episode_reward_terms["deaths"] += len(dying_agents)
        self.episode_global_reward += global_reward
        self._last_time = time
        self._last_global_uncertainty = global_uncertainty
        return local_rewards, global_reward, dying_agents

    ##### Tensors #####
    def _fill_observation(self, td: TensorDictBase, episode_ended: bool, closing=()) -> None:
        """
        Observation of every flagged slot, built by the drone that decides. When the
        episode ends, every slot alive gets its own observation, the next observation of
        the transition the orchestrator closes. The other slots stay at zero.
        """
        agents_td = td.get("agents")
        obs_td = agents_td.get("observation")
        mask = agents_td.get("mask")
        for agent in self._active_episode_agents():
            mask[agent.slot_index] = True

        requests = dict(self.flagged_requests)
        if episode_ended:
            requests = {agent.slot_index: self._own_request(agent) for agent in self._active_episode_agents()}

        for slot, request in requests.items():
            observation = self.protocol(request.decider_id).observe(request)
            for key, value in observation.items():
                obs_td.get(key)[slot] = torch.as_tensor(value, device=self.device)
            obs_td.get("decision_flag")[slot, 0] = not episode_ended

        if requests or closing or episode_ended:
            self._fill_global_state(td.get("global_state"))

    def _fill_global_state(self, global_td: TensorDictBase) -> None:
        L = self.large_map_size
        global_map = self.monitor.global_map()
        mean, std = global_map.mean(), global_map.std()
        normalized = np.clip((global_map - mean) / (std + 1e-4), -UNCERTAINTY_CLIP, UNCERTAINTY_CLIP)
        normalized = normalized.reshape(L, self.map_width // L, L, self.map_height // L).mean(axis=(1, 3))
        global_td.get("global_map").copy_(torch.as_tensor(normalized, dtype=torch.float32, device=self.device))

        half_side = self.map_width * self.distance_between_cells / 2
        drones = np.zeros((self.max_num_agents, 7), dtype=np.float32)
        charging = dead = 0
        for agent in self._existing_episode_agents():
            protocol = self.protocol(agent.node_id)
            position = protocol.drone_position if protocol.drone_position is not None else (0.0, 0.0, 0.0)
            drones[agent.slot_index] = (1.0, float(protocol.status != DroneStatus.DEAD),
                                        (position[0] + half_side) / (2 * half_side),
                                        (position[1] + half_side) / (2 * half_side),
                                        protocol.battery.battery_status,
                                        float(protocol.status == DroneStatus.GOING_TO_BASE),
                                        float(protocol.status == DroneStatus.CHARGING))
            charging += protocol.status == DroneStatus.CHARGING
            dead += protocol.status == DroneStatus.DEAD
        global_td.get("drones").copy_(torch.as_tensor(drones, device=self.device))
        n = self.number_of_drones
        global_td.get("swarm").copy_(torch.tensor(
            [min(float(mean), UNCERTAINTY_CLIP), charging / n, self.nc_charging_base / n, dead / n,
             self.current_time / self.config.max_episode_length], dtype=torch.float32, device=self.device))
        global_td.get("all_active").copy_(torch.tensor(
            [agent.exists and agent.active for agent in self.episode_agents], device=self.device))

    def _fill_rewards(self, td: TensorDictBase, local_rewards: dict, global_reward: float) -> None:
        agents_reward_t = td.get(("agents", "reward"))
        for slot, reward in local_rewards.items():
            agents_reward_t[slot, 0] = float(reward)
        td.get("global_reward")[0] = float(global_reward)

    def _fill_done(self, td: TensorDictBase, end_cause: EndCause, dying_agents: list[EpisodeAgentState]) -> None:
        """
        A drone that died in the step is terminated (its transition is terminal). When the
        episode ends on time the drones alive are truncated (their value is bootstrapped).
        Slots dead before, and slots without a drone, are done for bookkeeping only.
        """
        done = td.get(("agents", "done"))
        terminated = td.get(("agents", "terminated"))
        truncated = td.get(("agents", "truncated"))

        dying_slots = self._agent_slot_tensor(dying_agents)
        terminated.index_fill_(0, dying_slots, True)
        done.index_fill_(0, dying_slots, True)

        if end_cause != EndCause.NONE:
            active_slots = self._agent_slot_tensor(self._active_episode_agents())
            truncated.index_fill_(0, active_slots, True)
            done.index_fill_(0, active_slots, True)

        bookkeeping = [agent for agent in self.episode_agents
                       if (not agent.exists or not agent.active) and agent not in dying_agents]
        bookkeeping_slots = self._agent_slot_tensor(bookkeeping)
        truncated.index_fill_(0, bookkeeping_slots, True)
        done.index_fill_(0, bookkeeping_slots, True)

        episode_done = end_cause != EndCause.NONE
        td.set("done", torch.tensor([episode_done], device=self.device))
        td.set("terminated", torch.tensor([end_cause == EndCause.ALL_AGENTS_INACTIVE], device=self.device))
        td.set("truncated", torch.tensor([end_cause == EndCause.TIMEOUT], device=self.device))

    def episode_metrics(self, end_cause: EndCause = EndCause.NONE) -> dict:
        """Summary of the episode so far, the info at its end and what the worker reports."""
        monitor = self.monitor
        charging = np.array(monitor.charging_drones)
        return {
            "reward": self.episode_global_reward,
            "episode_duration": float(self.current_time),
            "cause": float(end_cause.value),
            "num_agents": float(self.number_of_drones),
            "mean_global_uncertainty": float(np.mean(monitor.uncertainty)) if monitor.uncertainty else float("nan"),
            "dead_drones": float(monitor.dead_drones[-1]) if monitor.dead_drones else 0.0,
            "time_above_nc": float(np.mean(charging > self.nc_charging_base)) if charging.size else 0.0,
            "cost": monitor.cost(),
            "decisions": float(sum(self.decisions.values())),
            **{f"reward_{name}": value for name, value in self.episode_reward_terms.items()},
        }

    def _fill_info(self, td: TensorDictBase, end_cause: EndCause, ended: bool) -> None:
        existing_agents = self._existing_episode_agents()
        if not ended or not existing_agents:
            return
        info_td = td.get(("agents", "info"))
        existing_slots = self._agent_slot_tensor(existing_agents)
        for key, value in self.episode_metrics(end_cause).items():
            if key in self._all_info_keys:
                info_td.get(key).index_fill_(0, existing_slots, value)

    def close(self, *, raise_if_closed: bool = True):
        super().close(raise_if_closed=raise_if_closed)
        self.finalize_simulation()
