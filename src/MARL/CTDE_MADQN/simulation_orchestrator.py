"""
SIMULATION ORCHESTRATOR

Wraps a MappingEnvironment and produces SMDP transitions: one per decision of an agent,
from the decision to its next one (or its death, or the end of the episode).

A transition of agent i:
    obs, action      the observation and action of the decision
    reward           the rewards of every step until the next decision, each one discounted
                     by discount_per_second ** (seconds since the decision)
    next_obs         the observation of the next decision
    duration         simulated seconds between the two decisions. The DQN target bootstraps
                     with discount_per_second ** duration, so a 10 s move to the next cell
                     and a 200 s trip to the base are discounted by the time they take
    done             True when the agent died before deciding again (terminal). At the end
                     of the episode the transition is closed with done False: the value of
                     the last observation is bootstrapped

CTDE: the reward of agent i in a step is its local reward + local_global_reward_ratio *
the global reward (see MappingEnvironment for both). The global state is stored too,
for a centralized critic.
"""

from dataclasses import dataclass
from typing import Callable
import torch
from tensordict import TensorDict


@dataclass
class PendingTransition:
    ##### Agent
    agent_state: TensorDict     # the per-agent slice of the observation at decision time
    agent_action: torch.Tensor  # shape (1,)
    ##### Global
    global_state: TensorDict   # the "global_state" subtree at decision time
    joint_action: torch.Tensor # shape (N, 1) — last-committed action per agent
    open_time: float           # simulated time of the decision
    ##### Agent
    agent_reward_sum: float = 0.0
    agent_n_sim_steps: int = 0  # how many environment steps this macro-action has lasted


class AsyncMARLOrchestrator:
    """
    `policy_fn(per_agent_obs_td) -> action_tensor` is the decision rule
    random, scripted, or a neural network. It only ever sees observations
    for agents that are actually making a decision this step.
    """

    def __init__(self, env, policy_fn: Callable[[TensorDict], torch.Tensor],
                 discount_per_second: float = 0.999, local_global_reward_ratio: float = 0.5,
                 collect_transitions: bool = True):
        self.env = env
        self.policy_fn = policy_fn
        self.N = env.max_num_agents
        self.DISCOUNT_PER_SECOND = discount_per_second
        self.LOCAL_GLOBAL_REWARD_RATIO = local_global_reward_ratio
        self.collect_transitions = collect_transitions

        self.pending_transition: list[PendingTransition | None] = [None] * self.N
        # Actions chosen at the last step, applied by the next one
        self.actions_to_apply: dict[int, torch.Tensor] = {}

        # The most recent action each agent committed to. Updated only on flag events.
        self.last_committed_action = torch.zeros(self.N, 1, dtype=torch.int64)

        # Output buffers
        self.transitions: list[TensorDict] = []
        self.episode_stats: list[dict] = []   # MappingEnvironment.episode_metrics of every episode ended

    @staticmethod
    def _read_flags(obs_td: TensorDict) -> torch.Tensor:
        """Return a (N,) bool tensor: True if the agent has a pending decision."""
        return obs_td["agents", "observation", "decision_flag"].squeeze(-1).bool()

    def _slice_agent_obs(self, obs_td: TensorDict, i: int) -> TensorDict:
        """Take just agent i's observation, cloned so the env can't mutate it."""
        obs = obs_td["agents", "observation"]
        return TensorDict(
            {key: value[i].clone() for key, value in obs.items() if key != "decision_flag"},
            batch_size=[],
            device=obs.device,
        )

    def _snapshot_global(self, obs_td: TensorDict) -> TensorDict:
        return obs_td["global_state"].clone()

    def agent_reward(self, local_reward: float, global_reward: float) -> float:
        """CTDE: the local reward plus a share of the reward of the swarm."""
        return local_reward + self.LOCAL_GLOBAL_REWARD_RATIO * global_reward

    def reset(self):
        td = self.env.reset()
        self.pending_transition = [None] * self.N
        self.actions_to_apply = {}
        self.last_committed_action.zero_()
        self._handle_flags(td)
        return td

    def step(self, td: TensorDict) -> TensorDict:
        # Random actions for unflagged agents (the env ignores them).
        td = self.env.rand_action(td)

        # Overwrite the rows of the agents that decided at the last step.
        action = td["agents", "action"]
        for i, new_action in self.actions_to_apply.items():
            action[i] = new_action
        self.actions_to_apply = {}

        # Step.
        td = self.env.step(td)
        now = self.env.current_time

        # Accumulate rewards into all pending transitions, discounted from their decision.
        rewards = td["next", "agents", "reward"].squeeze(-1)
        global_reward = td["next", "global_reward"].item()
        for i in range(self.N):
            p = self.pending_transition[i]
            if p is not None:
                discount = self.DISCOUNT_PER_SECOND ** (now - p.open_time)
                p.agent_reward_sum += discount * self.agent_reward(rewards[i].item(), global_reward)
                p.agent_n_sim_steps += 1

        terminated = td["next", "agents", "terminated"].squeeze(-1)

        # Episode end: the agents that died now are terminal, the others are bootstrapped.
        if td["next", "done"].item():
            for i in range(self.N):
                if self.pending_transition[i] is not None:
                    self._close_agent(i, td["next"], terminal=bool(terminated[i]))
            self.episode_stats.append(self.env.episode_metrics(self._end_cause(td["next"])))
            return td

        # Per-agent deaths.
        for i in range(self.N):
            if terminated[i] and self.pending_transition[i] is not None:
                self._close_agent(i, td["next"], terminal=True)

        # New flags at t+1.
        self._handle_flags(td["next"])

        return td

    @staticmethod
    def _end_cause(next_td: TensorDict):
        from env.environment import EndCause
        if next_td["terminated"].item():
            return EndCause.ALL_AGENTS_INACTIVE
        return EndCause.TIMEOUT

    ##### flag handling
    def _handle_flags(self, obs_td: TensorDict):
        flags = self._read_flags(obs_td)
        mask = obs_td["agents", "mask"]

        for i in range(self.N):
            if not mask[i] or not flags[i].item():
                continue

            if self.pending_transition[i] is not None:
                self._close_agent(i, obs_td, terminal=False)

            per_agent_obs = self._slice_agent_obs(obs_td, i)
            new_action = self.policy_fn(per_agent_obs).reshape(1).to(torch.int64)
            self.last_committed_action[i] = new_action
            self.actions_to_apply[i] = new_action

            if self.collect_transitions:
                self.pending_transition[i] = PendingTransition(
                    agent_state=per_agent_obs,
                    agent_action=new_action.clone(),
                    global_state=self._snapshot_global(obs_td),
                    joint_action=self.last_committed_action.clone(),
                    open_time=self.env.current_time,
                )

    def _close_agent(self, i: int, next_obs_td: TensorDict, terminal: bool):
        p = self.pending_transition[i]
        transition = TensorDict({
            # agent side
            "obs":         p.agent_state,
            "next_obs":    self._slice_agent_obs(next_obs_td, i),
            "action":      p.agent_action,
            "reward":      torch.tensor([p.agent_reward_sum], dtype=torch.float32),
            "done":        torch.tensor([terminal], dtype=torch.bool),
            "duration":    torch.tensor([self.env.current_time - p.open_time], dtype=torch.float32),
            "n_sim_steps": torch.tensor([p.agent_n_sim_steps], dtype=torch.long),
            "agent_idx":   torch.tensor([i], dtype=torch.long),
            # global side
            "global_obs":         p.global_state,
            "global_next_obs":    self._snapshot_global(next_obs_td),
            "joint_action":       p.joint_action,
        }, batch_size=[])
        self.transitions.append(transition)
        self.pending_transition[i] = None
