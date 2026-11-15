"""
Sanity check of the environment and of the transitions the training gets.

    python3 env_check.py                                   # 3 drones, 1500 s, random policy
    python3 env_check.py --drones 5 --duration 3000 --range 2000
    python3 env_check.py --policy analytical               # the analytical decisions as actions

It runs TorchRL's check_env_specs, then one episode through the orchestrator, the way
a training worker does, and checks the transitions:
    - one transition per decision, closed at the next decision, the death or the end;
    - every action valid in its observation, every observation with a valid action;
    - the transitions of the drones that died are terminal and carry the death penalty,
      the ones open at the end are not terminal (they are bootstrapped);
    - the duration is the time between the decision and the next one.
It prints the statistics of the transitions (durations, rewards) and of the episode.
Run it from this directory, with gradysim on the PYTHONPATH.
"""

import sys
import argparse

import numpy as np
import torch
from torchrl.envs.utils import check_env_specs

import config
from env.mapping_environment import MappingEnvironment
from simulation_orchestrator import AsyncMARLOrchestrator
from policy import RandomPolicy


class AnalyticalActions:
    """
    The action the analytical fitness function takes, given as the policy's action: the
    decisions go through the same path as the network's. The run is then the run of
    Analytical/swarm, as with decision_backend "analytical".
    """

    def __init__(self, env):
        self.env = env
        self._requests = None
        self.pending = []

    def __call__(self, per_agent_obs):
        ##### The orchestrator asks for the flagged slots of the step in slot order. The env #####
        ##### builds a new flagged_requests dict on every step.                              #####
        if self._requests is not self.env.flagged_requests:
            self._requests = self.env.flagged_requests
            self.pending = sorted(self._requests)
        slot = self.pending.pop(0)
        request = self.env.flagged_requests[slot]
        return torch.tensor([self.env.protocol(request.decider_id).analytical_action(request)])


def stats(values) -> str:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "none"
    return (f"mean {values.mean():.3f}, std {values.std():.3f}, min {values.min():.3f}, "
            f"max {values.max():.3f} (n={values.size})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check of the environment and its transitions")
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--duration", type=int, default=1500)
    parser.add_argument("--range", type=float, default=config.TRANSMISSION_RANGE, dest="transmission_range")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy", choices=("random", "analytical"), default="random")
    args = parser.parse_args()

    env_config = config.scenario_config(
        num_agents_choices=(args.drones,), seed=args.seed, max_episode_length=args.duration,
        communication_range=args.transmission_range,
        analytical_parameters=config.unpack_individual(config.BEST_INDIVIDUAL))

    print("Running check_env_specs ...")
    check_env_specs(MappingEnvironment(env_config))
    print("  OK\n")

    env = MappingEnvironment(env_config)
    if args.policy == "analytical":
        policy = AnalyticalActions(env)
    else:
        policy = RandomPolicy()
    orchestrator = AsyncMARLOrchestrator(env, policy)

    td = orchestrator.reset()
    steps = 0
    while True:
        td = orchestrator.step(td)
        steps += 1
        if td["next", "done"].item():
            break
        td = td["next"]
    env.close()

    transitions = orchestrator.transitions
    episode = orchestrator.episode_stats[-1]
    problems = []

    decisions = sum(env.decisions.values())
    ##### Every decision opens a transition, closed by the next decision, the death or the end #####
    if len(transitions) != decisions:
        problems.append(f"{len(transitions)} transitions for {decisions} decisions")

    terminal = [t for t in transitions if t["done"].item()]
    dead = int(episode["reward_deaths"])
    if len(terminal) != dead:
        problems.append(f"{len(terminal)} terminal transitions for {dead} deaths")
    death_penalty = config.REWARD.death_penalty
    for t in terminal:
        if t["reward"].item() > -0.5 * death_penalty:
            problems.append(f"terminal transition of agent {t['agent_idx'].item()} without the death penalty "
                            f"(reward {t['reward'].item():.2f})")

    for k, t in enumerate(transitions):
        action = t["action"].item()
        if not t["obs", "valid_actions"][action]:
            problems.append(f"transition {k}: invalid action {action}")
        if not t["done"].item() and not t["next_obs", "valid_actions"].any():
            problems.append(f"transition {k}: non-terminal with an empty next observation")
        if t["duration"].item() < 0:
            problems.append(f"transition {k}: negative duration")

    encounter = [t for t in transitions if t["obs", "encounter_state"][0] > 0]
    partner = [t for t in transitions if t["obs", "encounter_state"][5] > 0]
    base = [t for t in transitions if t["action"].item() == env.n_actions - 1]
    print(f"Episode: {args.drones} drones, {args.duration} s, {steps} steps, decisions {env.decisions}")
    print(f"Transitions: {len(transitions)} ({len(encounter)} of encounters, {len(partner)} chosen by the "
          f"other drone, {len(base)} to the base, {len(terminal)} terminal)")
    print(f"  duration, s: {stats([t['duration'].item() for t in transitions])}")
    print(f"  steps:       {stats([t['n_sim_steps'].item() for t in transitions])}")
    print(f"  reward:      {stats([t['reward'].item() for t in transitions])}")
    print(f"  reward of the transitions to the base: {stats([t['reward'].item() for t in base])}")
    print(f"  reward of the terminal transitions:    {stats([t['reward'].item() for t in terminal])}")
    print("Episode summary: " + ", ".join(f"{key} {value:.4g}" for key, value in episode.items()))
    print()
    for problem in problems:
        print(f"[FAIL] {problem}")
    print("[ok  ] transitions consistent" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
