"""
train.py — Double DQN with parameter sharing, collected by parallel workers.

    python3 train.py

Run it from this directory, with gradysim on the PYTHONPATH. The scenario is the one of
Analytical/swarm (config.py). Each worker runs an environment (env/mapping_environment.py)
whose episodes draw a swarm size from TRAINING_DRONE_COUNTS and a base position, as the
GA does, and sends back one transition per decision (simulation_orchestrator.py).

Sync lifecycle (per iteration):
    set_weights -> broadcast -> resume -> collect (returns with workers PAUSED)
    -> train -> next iteration

The checkpoints, the metrics (training_metrics.csv) and the episode summaries
(episode_metrics.csv) go to CHECKPOINT_DIR. Test a checkpoint with
    python3 main_test.py --policy nn --checkpoint checkpoints_energy/best.pt
"""
import os
import multiprocessing as mp
import torch
import time
import torch.nn.functional as F

import config
from env.mapping_environment import MappingEnvironment
from env.gradysim_environment.protocol import MAP_CHANNELS, AGENT_STATE_DIM, ENCOUNTER_STATE_DIM
from workers_orchestrator import WorkersOrchestrator
from replay_buffer import ReplayBuffer
from actor import Actor
from policy import DQNPolicy
from train_logs import TrainingLogger

##### Scenario of the episodes, see config.py #####
ALGORITHM_ITERATION_INTERVAL = 1.0   # longest environment step, simulated seconds
MAX_EPISODE_LENGTH = 4000            # simulated seconds, several battery cycles (a full battery lasts ~1200 s)
TRANSMISSION_RANGE = config.TRANSMISSION_RANGE
DRONE_COUNTS = config.TRAINING_DRONE_COUNTS
ACTION_MAP_SIZE = config.ACTION_MAP_SIZE
LARGE_MAP_SIZE = config.LARGE_MAP_SIZE

##### Network #####
MAP_FEATURE_DIM = 128
VECTOR_FEATURE_DIM = 64
HIDDEN_DIM = 256

##### Exploration #####
EPS_INIT = 1.0
EPS_DECAY = 0.9998
EPS_MIN = 0.1

##### Collection #####
N_WORKERS = 16
STEPS_PER_BATCH = 100
NUM_ITERATIONS = 20000
MIN_TRANSITIONS_PER_COLLECT = 100
COLLECT_TIMEOUT_S = 10.0
SYNC = False
NEW_BATCH_NEW_SIMULATION = False
BASE_SEED = 47

##### DQN #####
BATCH_SIZE = 256
BUFFERSIZE = 30000
VALUE_NETWORK_UPDATES_PER_ITERATION = 4
DISCOUNT_PER_SECOND = 0.999  # gamma per simulated second: rewards inside a transition and the bootstrap
LR = 1e-4
GRAD_CLIP = 1.0
TAU = 0.005

LOCAL_GLOBAL_REWARD_RATIO = 0.5 # CTDE: weight of the global reward added to the local one

CHECKPOINT_EVERY    = 500              # save every K iterations (set ≤ NUM_ITERATIONS)
CHECKPOINT_DIR      = "checkpoints_energy"  # the old checkpoints/ are of the network without energy
REWARD_WINDOW       = 100               # rolling-average window for "is it improving?"


def make_env():
    return MappingEnvironment(config.scenario_config(
        algorithm_iteration_interval=ALGORITHM_ITERATION_INTERVAL,
        num_agents_choices=DRONE_COUNTS,
        max_episode_length=MAX_EPISODE_LENGTH,
        communication_range=TRANSMISSION_RANGE,
    ))


def actor_kwargs():
    """The Actor of train.py, also used to load a checkpoint (main_test.py)."""
    return dict(
        action_dim=ACTION_MAP_SIZE * ACTION_MAP_SIZE + 1,
        map_channels=MAP_CHANNELS,
        large_map_size=LARGE_MAP_SIZE,
        small_map_size=ACTION_MAP_SIZE,
        agent_state_dim=AGENT_STATE_DIM,
        encounter_state_dim=ENCOUNTER_STATE_DIM,
        map_feature_dim=MAP_FEATURE_DIM,
        vector_feature_dim=VECTOR_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
    )


def make_policy():
    return DQNPolicy(eps_init=EPS_INIT, **actor_kwargs())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    replay_buffer = ReplayBuffer(buffer_size=BUFFERSIZE, batch_size=BATCH_SIZE, centralized_training=False)
    trainer_policy = make_policy().to(device)
    optimizer = torch.optim.Adam(trainer_policy.actor.parameters(), lr=LR)

    logger = TrainingLogger(
        checkpoint_dir=CHECKPOINT_DIR,
        reward_window=REWARD_WINDOW,
        checkpoint_every=CHECKPOINT_EVERY,
        live_plot=False,
        plot_every=1,
        )

    target_actor = Actor(**actor_kwargs()).to(device)
    target_actor.load_state_dict(trainer_policy.actor.state_dict())

    for p in target_actor.parameters():
        p.requires_grad_(False)
    target_actor.eval()

    with WorkersOrchestrator(
        num_workers=N_WORKERS,
        env_fn=make_env,
        policy_fn=make_policy,
        replay_buffer=replay_buffer,
        steps_per_batch=STEPS_PER_BATCH,
        base_seed=BASE_SEED,
        sync=SYNC,
        discount_per_second=DISCOUNT_PER_SECOND,
        local_global_reward_ratio=LOCAL_GLOBAL_REWARD_RATIO,
        new_batch_new_simulation=NEW_BATCH_NEW_SIMULATION,
    ) as orch:

        for it in range(NUM_ITERATIONS):
            # Same four-step rhythm on every iteration, including the first.
            orch.set_weights(trainer_policy.state_dict())
            orch.broadcast()
            orch.resume()
            new_count = orch.collect(
                min_new_transitions=MIN_TRANSITIONS_PER_COLLECT,
                timeout=COLLECT_TIMEOUT_S,
            )
            logger.log_episodes(it, orch.episode_stats)
            orch.episode_stats.clear()

            if SYNC:
                # stop the execution for 100 ms
                # This is necessary to give time to worker to stop before starting a new bunch of step iterations
                time.sleep(0.1)

            batch = None
            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(VALUE_NETWORK_UPDATES_PER_ITERATION):
                    batch = replay_buffer.sample()

                    obs = batch["obs"].to(device)
                    next_obs = batch["next_obs"].to(device)
                    action = batch["action"].to(device)
                    reward = batch["reward"].float().to(device)
                    done = batch["done"].float().to(device)
                    duration = batch["duration"].float().to(device)

                    with torch.no_grad():
                        #### Double DQN target calculation
                        next_q_value_online = trainer_policy.actor(next_obs)       # (B, action_dim)
                        a_next = next_q_value_online.argmax(dim=-1, keepdim=True)  # (B, 1)
                        next_q_target = target_actor(next_obs).gather(-1, a_next) #  B, 1)
                        # A dead drone's next_obs has no valid action, so its Q is -inf,
                        # and -inf * (1 - done) = -inf * 0 = nan. Zero it first.
                        next_q_target = next_q_target.masked_fill(done.bool(), 0.0)

                        ##### SMDP: the next decision comes `duration` seconds later #####
                        y = reward + (DISCOUNT_PER_SECOND ** duration) * next_q_target * (1.0 - done)

                    q_pred_all = trainer_policy.actor(obs)  # (B, action_dim)
                    q_pred = q_pred_all.gather(-1, action)  # (B, 1)

                    # SmoothL1 (Huber) is the DQN convention — more robust to outliers than MSE
                    loss = F.smooth_l1_loss(q_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    # For stability:
                    torch.nn.utils.clip_grad_norm_(trainer_policy.actor.parameters(), max_norm=GRAD_CLIP)
                    optimizer.step()

                    # soft update
                    for pt, po in zip(target_actor.parameters(), trainer_policy.actor.parameters()):
                        pt.data.mul_(1 - TAU).add_(TAU * po.data)

                    logger.log_update(
                        loss=loss.item(),
                        q_values_all=q_pred_all,
                        td_error=(y - q_pred).mean().item(),
                    )

                # Update EPS
                trainer_policy.update_epsilon(max(trainer_policy.eps.item() * EPS_DECAY, EPS_MIN))

                logger.log_iteration(
                    it=it,
                    new_count=new_count,
                    buffer_size=len(replay_buffer),
                    eps=trainer_policy.eps.item(),
                    reward_sample=batch["reward"] if batch is not None else None
                    )

                logger.maybe_checkpoint(it, trainer_policy.actor, target_actor, optimizer,
                                        eps=trainer_policy.eps.item())

        logger.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
