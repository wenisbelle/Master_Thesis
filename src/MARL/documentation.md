# MARL systems: architecture and changes

`src/MARL/CTDE_MADQN` and `src/MARL/Independent_MADQN`, updated on 2026-11-14.

## 1. Summary

The MARL systems now run **the protocol, the scenario and the constants of
`Analytical/swarm`**. The only difference is who chooses the destinations: a neural
network trained with RL instead of the fitness function.

Before this update, the MARL protocol was the old mapping-only one:
- a 50x50 map;
- the simple `MobilityHandler` with `GotoCoords`;
- an energy model that no decision used;
- no charging base, no charging and no battery estimation;
- deaths only by a random probability;
- estimated positions of the other drones, with timeouts (`drone_states`,
  `BROADCAST_DESTINATION_MESSAGE`, `NORMALIZE_TIME`).

All of that is gone (see section 11).

What is the same as `Analytical/swarm` (copied):
- the 100x100 map of 20 m cells, the uncertainty growth (0.001 per cell every 1 s) and
  initial uncertainty 1.0;
- the camera, the gossip of the map (base64), the heartbeat;
- the encounter rule: the drone with the highest id decides for both, at most once every
  10 s per pair;
- the `DynamicVelocity` mobility at 15 m/s and 3 m/s², the braking approach to the base;
- the gradysim `BatteryPowerPlugin` with random wind, and the death below 10% battery;
- the charging base: 100 s recharge, `CHARGING_NOTIFICATION`, `DEAD_NOTIFICATION`;
- the battery table of the swarm (`swarm_battery_status`) and its estimation
  (`estimate_drone_battery`, `aproximate_battery_consumption_model`);
- Nc = int(n/3); the base drawn with σ = 333 m during training; take-off from the base;
- the per-drone random streams seeded by (seed, id).

**Verified.** Replaying the analytical decisions through the MARL code gives the
analytical run **bit for bit**: every 1 s sample of global uncertainty, dead drones and
drones charging, every encounter, the final state of every drone, the cost, and even the
recording that the plots read. This holds for 3 to 10 drones, 100 to 2000 m and up to
10,000 s, with recharges, congestion and deaths (section 10).

**Reward.** A positive reward for the uncertainty removed, minus a penalty when a drone
dies, minus the penalty for more than Nc drones charging (the same formula as the
analytical cost). Section 6 has the details.

**Testing.** You test it the same way as the analytical system:
- `main_test.py`: same report, checks, recording and plots;
- `multiple_runs.py`: same run files; `Analytical/plot_multiple_runs.py` now draws them
  next to the analytical runs.

`--policy analytical` runs the analytical protocol through the MARL code, as the baseline
on the same scenario.

**CTDE vs Independent.** The two folders are identical except for one function,
`AsyncMARLOrchestrator.agent_reward` in `simulation_orchestrator.py` (and the matching
comment in `train.py`). CTDE uses local + 0.5 × global reward. Independent uses the local
reward only.

## 2. Files (the same in both folders)

| file | what it is |
|---|---|
| `config.py` | Scenario constants, copied from `Analytical/swarm/main.py`. Also the analytical cost weights, `BEST_INDIVIDUAL`, the action/map sizes and the reward weights. `scenario_config()` builds the environment configuration. |
| `env/gradysim_environment/protocol.py` | The drone protocol: a copy of `Analytical/swarm/protocol.py` where the decisions are requested from outside (section 3), plus `observe()` (section 5). |
| `env/gradysim_environment/fitness.py`, `visualization.py` | Byte-identical copies of the analytical ones. The fitness function is only used to replay the analytical decisions. |
| `env/base_gradys_env.py` | Steps the simulation event by event and stops at decisions (section 3). |
| `env/mapping_environment.py` | The TorchRL environment: specs, reset/step, observations, rewards, done flags. `MappingEnvironmentConfig`, `RewardWeights`, `sample_base_position`. |
| `env/swarm_monitor.py` | `SwarmMonitor`: the `GlobalMapMonitor` of `Analytical/swarm/main.py` (global map, dead, charging, the analytical cost), plus the congestion part of the reward. |
| `env/metrics.py`, `env/environment.py` | Episode summary keys, `EndCause`. |
| `actor.py` | The Q network (section 8). |
| `policy.py` | `RandomPolicy` (uniform over valid actions), `DQNPolicy` (ε-greedy), `load_greedy_policy(checkpoint)`. |
| `simulation_orchestrator.py` | Turns environment steps into SMDP transitions, one per decision (section 7). **The only file that differs between CTDE and Independent.** |
| `worker.py`, `workers_orchestrator.py`, `replay_buffer.py` | Parallel collection, as before. Each worker also sends the summary of every episode it finishes. |
| `train.py`, `train_logs.py` | Double DQN training and its logs/checkpoints (section 8). |
| `main_test.py` | Single test run, the counterpart of `Analytical/swarm/main_test.py` (section 9). |
| `multiple_runs.py` | Many seeded runs, the counterpart of `Analytical/multiple_runs.py` (section 9). |
| `recorder.py`, `estimation_metrics.py`, `plots/` | Copies of the analytical ones. In `recorder.py`, `PARAMETERS` reads Nc and the minimum margin from the protocol instead of the fitness function, which a network run does not have. |
| `check_equivalence.py` | Regression gate: the MARL run against `Analytical/swarm`, bit for bit (section 10). |
| `env_check.py` | Sanity check of the specs and of the transitions the training gets. |
| `checkpoints/` | **Old** checkpoints of the network without energy. They cannot be loaded by the new network (different inputs and outputs). The new training writes to `checkpoints_energy/`. |

Run everything from inside the system folder, with gradysim on the PYTHONPATH (it is in
the container): `cd /Thesis/src/MARL/CTDE_MADQN && python3 main_test.py ...`.

## 3. How a decision flows

The analytical protocol chooses a destination in two methods. Here each one becomes a
`DecisionRequest`:

| analytical | MARL (`DecisionKind`) | when |
|---|---|---|
| `internal_mobility_command()` → `choose_one_cell` | `OWN` | The drone reached its destination (`trigger="arrival"`), finished a recharge (`"recharge_done"`), or heard a new drone charging (`"charging_notification"`, also while GOING_TO_BASE). |
| `external_mobility_command()` → `choose_two_cells` (highest id) | `ENCOUNTER`, then `ENCOUNTER_PARTNER` | Gossip from a lower id, not in the last 10 s. |

```
event (timer / packet) in the protocol
   └─ protocol.request_decision(request) ──► decision_requests (list shared with the env)
env.step_simulation(): stops right after that event (no simulated time passes)
env returns: slot flagged (decision_flag), observation = decider.observe(request)
orchestrator: action = policy(observation); closes the agent's previous transition, opens a new one
next env.step(action): decider.apply_decision(request, action)
   OWN / ENCOUNTER: sets status + goto, sends the velocity command (as the analytical code does)
   ENCOUNTER: appends ENCOUNTER_PARTNER for the other drone → the env returns again,
              still at the same time, with the other drone's slot flagged
   ENCOUNTER_PARTNER: the decider sends SHARE_GOTO_POSITION (destination + status), as before
the simulation goes on until the next decision, the next 1 s boundary, or the end
```

Points that matter:
- **No simulated delay.** gradysim mobility commands only set a desired velocity (they
  schedule no event), and no handler acts in `after_simulation_step`. So applying the
  action after the event, before the next one, gives the same run as deciding inside the
  event. The old version applied the decision at the next 1 s tick of the environment.
  With the velocity commands of `DynamicVelocity`, the drone would keep flying during
  that wait.
- **No injected events.** gradysim orders events of the same instant by their place in
  the heap, so an extra event changes the run. The old environment scheduled one every
  iteration. `base_gradys_env.py` now steps exactly like `run_stepped_simulation`, and
  samples the swarm on the same 1 s grid.
- **Encounters keep the analytical rule.** The highest id decides for both, sequentially:
  first its own destination, then the other's, knowing its own choice
  (`encounter_state`). The other drone's observation is built by the decider, from the
  decider's merged map and battery table, which is the information `choose_two_cells` has.
  That observation is also the one stored in the other drone's transition.
- With `decision_backend="analytical"`, the environment takes the decisions itself with
  `protocol.analytical_action()`, which is the fitness function with the same inputs as
  the analytical code. It encodes the result as an action index and applies it through
  the same path. `env_check.AnalyticalActions` does the same as a policy, through the
  orchestrator.

## 4. Action space

`ACTION_MAP_SIZE² + 1 = 442` discrete actions:
- index `row*21 + col` is the cell `(i + row − 10, j + col − 10)`, where `(i, j)` is the
  drone's cell (`position_to_cell`, the same rounding as the fitness function);
- index `441` is the base. Choosing it sets the status to GOING_TO_BASE and the
  destination to the base cell centre, as `choose_one_cell` does.

The fitness function chooses among the cells `[i−10, i+10) × [j−10, j+10)`, so the 21x21
window contains every analytical choice. `valid_actions` masks:
- the cells outside the map;
- the drone's own cell (the analytical candidates exclude it too).

The base is always valid. The environment raises an error if an invalid action arrives.
The Q network puts −inf on invalid actions, and the random/ε exploration only draws
valid ones.

## 5. Observation (`protocol.observe`)

| key | shape | content |
|---|---|---|
| `large_map_patch` | (3, 50, 50) | Whole map, averaged in 2x2 blocks. See the channels below. |
| `small_map_patch` | (3, 21, 21) | The action window at full resolution: `[c, row, col]` is the cell of action `row*21+col`. Off-map cells: uncertainty −5, energy −1. |
| `agent_state` | (15,) | `x, y` ∈ [0,1]; `base_dx, base_dy` (base − position)/2000 m; `battery`; `energy_to_base`; `safe_battery` = battery − energy_to_base − 0.20; `going_to_base`; map mean and std; **swarm estimate**: estimated mean battery of the other drones alive, estimated number of them charging / N, Nc / N, known dead / N; N / 10. |
| `encounter_state` | (9,) | Zeros for `OWN`. Otherwise: 1; partner offset; partner battery (as the decider knows it); partner energy to base; whether the partner's choice is known (`ENCOUNTER_PARTNER`); the partner's chosen destination offset; whether the partner chose the base. |
| `valid_actions` | (442,) | Section 4. |
| `decision_flag` | (1,) | The agent decides in this step. The actor does not use it. |

The three map channels are:
- **uncertainty**, normalized (u − mean)/std and clipped to ±5, as before;
- **distance** to the drone, divided by the map diagonal;
- **energy margin**: battery − energy(drone → cell → base) − MIN_CHARGE_MARGIN, clipped to
  [−1, 1]. This is the R of `energy_effect`, with the fixed minimum margin. It also
  encodes where the base is.

The swarm terms are the ones of the analytical protocol: `estimate_drone_battery` over
the decider's `swarm_battery_status`. For `OWN` and `ENCOUNTER` they equal
`aproximate_battery_consumption_model`; this was checked on 1,822 decisions. For
`ENCOUNTER_PARTNER` they are computed from the partner's point of view: "others" are the
other drones, including the decider, with its live battery.

The global state is also stored in the transitions, for a future centralized critic. It
holds the global map (50x50, normalized), per drone `[exists, alive, x, y, battery,
going_to_base, charging]`, and the swarm values `[mean uncertainty, charging/N, Nc/N,
dead/N, time]`. The DQN does not use it.

## 6. Reward

The terms are computed per environment step, for every drone alive at its start
(`mapping_environment._compute_rewards`, weights in `config.REWARD`):

```
local_i = s_u · uncertainty removed from drone i's own map
          − its share of the congestion penalty (if it is CHARGING while more than Nc are)
          − death_penalty                      (on the step drone i dies: its last transition)
global  = s_u · uncertainty removed from the global map / N
          − congestion penalty of the swarm
          − death_penalty · (drones that died in the step)
```

The agent's reward is `local_i + 0.5·global` in **CTDE** and `local_i` in
**Independent**.

- **Uncertainty removed** = decrease of the map + the growth every cell had in the step
  (0.001 per cell per second). It is what the cameras took away (for the own map, also
  what the gossip merged in). It is ≥ 0, so being alive is never worse than being dead.
  Without the growth term, a 100x100 map grows by 10 per second and the reward would be
  almost always negative, which makes dying attractive.
- **Congestion** = `K·(exp(β·excess/N) − 1)` per second, with excess = drones CHARGING −
  Nc. This is the formula of the analytical cost (`congestion_penalty_rate`), sampled on
  the same 1 s grid. The swarm pays it, and the drones charging split it.
- **Death**: a one-off penalty on the step the drone dies. It ends the drone's
  transitions (terminal). It is the RL counterpart of the analytical cost per dead drone
  per second, which a terminal transition cannot carry.
- Defaults: `s_u = 0.01`, `death_penalty = 20`, `K = 0.1`, `β = 5`. The old per-step
  ([−1, 1]) and per-transition ([−5, 5]) reward clipping was removed: it would flatten
  the death penalty to the size of a normal step.

Measured on the analytical policy (3 drones, 200 m, 2,500 s, CTDE):
- 489 transitions, reward mean 1.18 (min −12.4, max 52.8);
- over the episode: local uncertainty term 525, global 62, congestion −13.7.

With the random policy, the death transitions are ≈ −36 in CTDE (−20 local, −10 from
the global term, plus the rest) and ≈ −20 in Independent.

The analytical cost (`UNCERTAINTY_WEIGHT·∫U + DEAD_DRONE_PENALTY·∫dead + ∫congestion`,
same weights) is computed on every run by `SwarmMonitor.cost()`. It is reported by the
tests and logged per training episode, so RL and analytical results are on the same
scale.

## 7. Transitions (`simulation_orchestrator.py`)

There is one transition per decision of an agent, from that decision to its next one:

- `reward` = Σ over the steps of `discount_per_second^(t_step − t_decision) · r_step`.
- `duration` = simulated seconds between the two decisions. The DQN target bootstraps
  with `DISCOUNT_PER_SECOND ** duration` (an SMDP). A 10 s move and a 200 s trip to the
  base are discounted by the time they take. Before, the in-transition rewards were
  discounted per step but the bootstrap used a fixed γ per decision.
- `done` = True only when the agent died before deciding again. When the episode ends,
  the open transitions are closed with `done=False` and the agent's own observation at
  that moment (the value is bootstrapped, a time limit is not a terminal state).
  A dead drone's `next_obs` has no valid action, so the target network gives it −inf.
  `train.py` sets the bootstrap of terminal transitions to 0 before multiplying, because
  `−inf · (1 − done)` = `−inf · 0` = nan, which spreads to the weights.
- The global state and the joint action are stored, as before.

`DISCOUNT_PER_SECOND = 0.999` gives a horizon of ≈ 1000 s, about one battery cycle (a
full battery reaches the 10% threshold after ~1,230 s).

## 8. Network and training

The `Actor` has the same structure as before:
- `DualMapEncoder`: the large map is convolved down to 21x21; the small patch is
  convolved at 21x21; the two are concatenated and jointly convolved;
- `VectorEncoder` (MLP) over `agent_state + encounter_state` (24 inputs);
- a 2-layer trunk (256) and a Q head with 442 outputs, −inf on invalid actions.

Two changes: 3 input channels, and a stride-2 convolution at the end of the joint head
(the window grew from 11 to 21 cells, and the flattened features would have made a 7M
parameter projection). 1.52 M parameters in total. The checkpoints store
`actor_kwargs`, so a test rebuilds the exact network that was trained.

`train.py` is the same Double DQN loop as before, with these changes:
- the scenario comes from `config.py`;
- each episode draws a swarm size from (3, 5, 7, 10) and a base position from its seed,
  as the GA does;
- `MAX_EPISODE_LENGTH = 4000` s (several battery cycles);
- the time-based discount (section 7).

Outputs go to `checkpoints_energy/`:
- `training_metrics.csv`: per iteration, as before, now with a plain-number `eps` column;
- `episode_metrics.csv` (new): one row per finished episode with swarm size, mean global
  uncertainty, dead drones, share of time above Nc, analytical cost, reward and its terms;
- `best.pt`: now selected on the rolling mean of the **episode** reward once episodes
  exist. The sampled batch is a mix of old transitions.

A 40-iteration smoke run (2 workers, 400 s episodes) went through the whole loop:
workers, queue, replay buffer, updates, checkpoints and both CSVs.

Cost per worker (single thread), with the network, for 2,000 simulated seconds:
- ≈ 25 s with 5 drones at 200 m;
- ≈ 200 s with 10 drones at 2000 m (≈ 10,000 decisions: every pair in range meets every
  10 s).

## 9. Testing, as with the analytical system

From the system folder (e.g. `src/MARL/CTDE_MADQN`):

```bash
# one run, report + checks + recording (logs/test_recording.pkl), exit status 0 when every check passes
python3 main_test.py --checkpoint checkpoints_energy/best.pt --drones 5 --duration 10000 --seed 1
python3 main_test.py --policy analytical --drones 5 --seed 1          # the analytical baseline, same scenario
python3 main_test.py --policy random --drones 5 --seed 1              # lower bound
python3 main_test.py --checkpoint ... --transmission-range 500 --base-position 400 -200 --logs_directory logs_500

# plots, from plots/ (the same scripts as Analytical/swarm/plots)
cd plots && python3 plot_drones.py --directory logs_500   # also plot_metrics, plot_more_metrics, plot_states,
                                                          # plot_decisions, plot_estimation, plot_returns

# many runs, run k seeded with seed+k: same base and wind as run k of Analytical/multiple_runs.py
python3 multiple_runs.py --checkpoint checkpoints_energy/best.pt --runs 20 --drones 5 --range 200
# then, from src, compare with the analytical runs of the same seeds (paired Wilcoxon on the share above Nc):
python3 Analytical/multiple_runs.py --system swarm --runs 20 --drones 5 --range 200
python3 Analytical/plot_multiple_runs.py --system swarm MARL/CTDE_MADQN MARL/Independent_MADQN --range 200 --drones 5

# checks
python3 check_equivalence.py                     # MARL vs Analytical/swarm, bit for bit (exit 0 = identical)
python3 env_check.py --drones 5 --duration 3000  # specs + transitions consistency
```

`main_test.py` prints the same report as the analytical one:
- per-drone table, status timeline, global map, encounters;
- the analytical cost, with its terms;
- the battery estimation errors and the returns to the base;
- the same six checks.

It adds the decisions by kind and the reward of the swarm. With `--policy analytical`
and the same `--seed`/`--base-position`/individual, its report and its recording are
identical to `Analytical/swarm/main_test.py`.

To let `plot_multiple_runs.py` read the MARL runs, `Analytical/multiple_runs.py` gained
`MARL_SYSTEMS` (read only: it does not run them) and `system_dir` resolves them under
`src/MARL`. `Analytical/plot_multiple_runs.py` gained their styles: Okabe-Ito orange and
sky blue, with their own dash pattern and marker.

## 10. Verification done (2026-11-14)

**Bit-for-bit equivalence** (`check_equivalence.py`). These quantities are identical:
- every 1 s sample of global uncertainty, dead drones, drones charging and unvisited
  cells;
- the number of encounters;
- the final state of every drone;
- the cost.

"Both" in the "paths" column means the environment backend and the policy path through
the orchestrator; the policy path is the network's.

| seed | drones | range | duration | base | encounters | decisions own/enc/partner | max charging | dead | paths |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 2000 m | 600 s | (120, 280) | 155 | 20/155/155 | 0 | 0 | backend |
| 1 | 3 | 200 m | 4000 s | (120, 280) | 58 | 655/58/58 | 2 | 0 | backend |
| 1 | 5 | 200 m | 4000 s | (120, 280) | 139 | 1084/139/139 | 3 | 0 | backend |
| 2 | 3 | 2000 m | 4000 s | (60, −180) | 775 | 272/775/775 | 1 | 0 | backend |
| 2 | 5 | 2000 m | 4000 s | (60, −180) | 2836 | 545/2836/2836 | 3 | 0 | backend |
| 3 | 10 | 500 m | 10000 s | (680, −860) | 7495 | 6089/7495/7495 | 4 | 0 | backend |
| 4 | 7 | 100 m | 10000 s | (0, 0) | 393 | 3975/393/393 | 4 | 0 | backend |
| 1 | 3 | 200 m | 2500 s | (120, 280) | 49 | 391/49/49 | 2 | 0 | both |
| 5 | 3 | 2000 m | 2000 s | (−260, −440), κ = 0 | 287 | 90/287/287 | 0 | **3** | both |
| 5 | 5 | 2000 m | 2000 s | (−260, −440), κ = 0 | 951 | 133/951/951 | 0 | **5** | both |
| 9 | 5 | 500 m | 2000 s | (−260, 80) | 323 | 404/323/323 | 4 | 0 | both (Independent folder) |

κ = 0 is `BEST_INDIVIDUAL` with `charging_base_multiplier = 0`: the drones never go
back, so death, DEAD notifications and the dead cost are exercised.

The other checks:
- **Every analytical decision fits in the action space.** The environment asserts that
  each action is valid; tens of thousands of analytical decisions passed.
- **main_test.** `main_test.py --policy analytical` and `Analytical/swarm/main_test.py`
  (seed 7, 3 drones, 200 m, 2,500 s, base (200, −400)) print the same report, and their
  recordings have 0 differences (every array, decision and return event).
- **multiple_runs.** MARL and analytical `multiple_runs.py`, seeds 10–12, give identical
  summaries.
- **Observations.** 1,822 decisions checked (704 own, 559 encounter, 559 partner):
  - the swarm estimate equals `aproximate_battery_consumption_model`;
  - the battery and the action-window patch are correct;
  - `ENCOUNTER_PARTNER` carries the decider's chosen destination.
- **Transitions** (`env_check.py`): one per decision; every action valid; the
  transitions of the drones that died are terminal and carry the death penalty; the
  ones open at the end are not terminal; `check_env_specs` passes.
- **Plots.** All 7 plot scripts run on recordings of the network, random and
  analytical policies.

**Policy behaviour, untrained** (5 drones, 200 m, 3,000 s, seed 3):
- random valid actions: 46% coverage; every drone dies at 1,200–1,235 s, since it almost
  never picks the base;
- untrained network: the drones cluster (561 encounters), 17% coverage, every drone
  dead.

Both are as expected without training: the energy behaviour has to be learned.

## 11. What was removed

- `drone_states` (estimated positions of the other drones + time of last update),
  `BROADCAST_DESTINATION_MESSAGE`, `get_estimated_drone_destinations`, `NORMALIZE_TIME`,
  and the `estimated_positions_and_time` observation.
- The distance penalties of the reward: the immediate one, against the estimated
  positions, and the global one, between all drones vs the communication range.
- The flag / `mobility_command_buffer` / `waiting_for_destination` mechanism, replaced by
  decision requests.
- `agent_death_probability`, the random deaths. Drones now die from the battery.
- The old energy model (`energy.py`) and `MobilityHandler`/`GotoCoords`. The battery is
  the gradysim plugin and the mobility is `DynamicVelocity`, as in the analytical system.
- The random initial maps (four distributions). The map starts at 1.0, as in the
  analytical system.
- The reward clipping. In Independent, a bug added a constant +0.5 to every step reward.
- Unused files: `utils.py` (TanhDelta), the empty `learning_algorithm.py`, the broken
  smoke tests `mapping_environment_test.py` / `mapping_environment_v0.py` (replaced by
  `env_check.py`), and `env/gradysim_environment/execution.py`.

## 12. Notes for the training work

- **Exploring the base.** Uniform ε exploration picks the base 1 time in 442. Early on,
  the drones will mostly die at ~1,230 s and rarely see a recharge. Options, from least
  to most invasive:
  - bias the ε draw towards the base when `safe_battery` is low;
  - pre-fill the replay buffer with **analytical demonstrations**: `env_check.AnalyticalActions`
    as the policy of an orchestrator with `collect_transitions=True` produces expert
    transitions in exactly the training format (DQfD-style, or a behaviour-cloning
    warm start);
  - an optional energy shield that masks cells with a negative energy margin.
- **Local reward and gossip.** The own-map term also counts the uncertainty merged from
  other drones' maps, so meeting is rewarded (it was like this before). The global term
  counts only camera coverage. This matters most for Independent.
- **Reward scale.** The defaults are a first guess (section 6), and nothing has been
  tuned yet. `episode_metrics.csv` logs each term, so the balance can be measured as it
  was done for the GA cost.
- **One training per range**, like the GA logbooks per range: `TRANSMISSION_RANGE` in
  `train.py`. Long ranges cost more: at 2000 m every pair of drones meets every 10 s, so
  a 10-drone episode takes ~8x longer than a 5-drone episode at 200 m (section 8).
- **Memory.** One transition (obs + next obs + global) is ≈ 90 KB, so
  `BUFFERSIZE = 30000` ≈ 2.7 GB of RAM (the machine has 15 GB).
- **GPU.** The host driver is 550 (CUDA 12.4). The default PyPI torch is built for
  CUDA 13 and needs driver ≥ 580: `nvidia-smi` works but `torch.cuda.is_available()` is
  False, and training silently runs on the CPU. The container therefore has
  `torch 2.11.0+cu126` (installed on 2026-11-14 from the PyTorch cu126 index; the
  `dockerfile` does the same). Check with
  `python3 -c "import torch; print(torch.cuda.is_available())"`; `train.py` prints the
  device it uses at start. On the RTX 3060 one update (batch 256) takes 68 ms on the
  GPU versus 1.3 s on the CPU, with 549 MiB of GPU memory, so the bottleneck is the
  CPU workers simulating, not the learner. `torchrl 0.15.0`/`tensordict 0.15.6` were
  installed the same day and the `dockerfile` installs them. TorchRL warns that torch
  < 2.13 is deprecated from v0.17; it works with 0.15.
- **Inherited from the analytical protocol.** The battery keeps draining at hover while
  CHARGING (the plugin is not shut down at the base). Fix it in `Analytical/swarm` first,
  then copy the fix here (next point).

## 13. Keeping MARL in sync with Analytical/swarm

These files are copies, as `expanding_testing` is:
- `protocol.py`: everything except the decision part (section 3) and `observe`;
- `fitness.py`, `visualization.py`, `estimation_metrics.py`, `plots/`;
- `recorder.py`: one change, see the `PARAMETERS` row in section 2;
- the constants in `config.py`.

After changing `Analytical/swarm`:
1. apply the same change here, in both folders;
2. run `python3 check_equivalence.py`. It compares the constants and runs both systems;
   exit 0 means bit-identical.

An added node or event breaks the bit-for-bit comparison (see `expanding_testing`).
The environment adds neither.
