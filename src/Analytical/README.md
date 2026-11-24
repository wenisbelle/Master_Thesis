# Analytical: the swarm and egocentric coordination protocols

A swarm of drones keeps a map up to date. Every cell of the map loses information
as time passes (its uncertainty grows), and a drone flying over it brings it back
to zero. The drones recharge at a base. The two systems here share the scenario,
the mapping, the cost and the GA. They differ only in **how a drone decides when
to recharge**:

- **`swarm/`**: the drone also weighs the battery of the others and how many of
  them it believes are charging. It estimates both from what it hears in encounters.
- **`egocentric/`**: the drone only looks at its own battery.

[`expanding_testing/`](expanding_testing/README.md) is a copy of both with any
number of bases. It has its own README.

[`centralized/`](centralized/README.md) is the reference both are measured against:
the swarm drone with perfect information and a central charging scheduler. See
[Centralized](#centralized).

[`naive/`](naive/README.md) is the floor: the egocentric drone with no energy model,
going back to the base below 30% of battery. See [Naive](#naive).

This file collects the details that are otherwise only in the code comments, the
script docstrings and the log files. The source is the code as of 2026-11-15. When
the two disagree, the code is right and this file is out of date.

## Contents

| Path | What it is |
|---|---|
| `swarm/`, `egocentric/` | One system each. `main.py` holds the scenario and the GA (`--mode train`). `main_test.py` runs one recorded simulation with a report. Also `protocol.py`, `fitness.py` (the decisions), `recorder.py`, `estimation_metrics.py` and `plots/` (plots of one recorded run) |
| `centralized/` | The same files for the centralized reference, plus `world.py` (global map, true state, charging scheduler) and `check_centralized.py`. See [Centralized](#centralized) |
| `charging_station.py` | The pads and the waiting room of a recharge base, used by every system (since 2026-11-24). See [Battery, recharge and death](#battery-recharge-and-death) |
| `naive/` | The same files for the naive baseline (no energy model, back to the base below 30%). See [Naive](#naive) |
| `<system>/ga_logs/` | GA logbooks, one per communication range. The `Parameters:` line is the tuned individual |
| `<system>/test_logs/<range>/<drones>/` | Single test runs, written by `all_tests_distance.py` |
| `<system>/multiple_runs/<range>/<drones>/` | Many runs of one scenario, written by `multiple_runs.py` |
| `all_optimizations_distance.py` | Runs the GA once per communication range |
| `all_tests_distance.py` | Runs `main_test.py` with the tuned individual of every range and swarm size |
| `comparing_swarm_egocentric.py` | Swarm against egocentric on one test run |
| `plot_ga_cost.py` | GA history of one range: the best cost per generation and its terms |
| `multiple_runs.py` | Runs one system many times, with the bases drawn again on every run |
| `run_all_tests_with_multiple_runs.py` | `multiple_runs.py` for every tuned range |
| `plot_multiple_runs.py` | The figures over many runs: uncertainty, charging, time between visits |
| `estimation_decision.md` | Plan and pilot results: how good the swarm battery estimate is |
| `waiting_room.md` | The hard base capacity and the waiting room of 2026-11-24: what changed, why, and where in the code (swarm first) |
| `variance_explained.md` | What `base_variance` and `alpha_variance_modifier` do (in Portuguese) |

## Running

Run the scripts from `src/`. gradysim has to be importable. In the container
(`thesis_container`, where `src/` is `/Thesis/src`) it already is. On the host:

```
cd src
export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen
```

Every script can be started from any directory, and `--help` lists its options.

Timings on this machine (16 cores):
- A 10,000 s run of 3 drones, alone, takes about 1 min.
- Bigger swarms, and many runs at once, are much slower. On 2026-11-15, 1,500 s of
  10 drones on 2 bases took 4–4.5 min per run, with 8 runs at once.

File sizes:
- A test run with recording (`main_test.py`) writes about 160 MB for 3 drones.
- A `multiple_runs.py` run file is under 1 MB.

## Workflow

```
python3 Analytical/all_optimizations_distance.py --system swarm        # 1. tune: one GA per range
#    then move <system>/ga_logbook_<range>.txt into <system>/ga_logs/
python3 Analytical/all_tests_distance.py --system swarm                # 2. one recorded run per range and size
python3 Analytical/multiple_runs.py --system swarm --runs 20 --range 200 --drones 10    # 3. many runs
python3 Analytical/plot_multiple_runs.py --system swarm egocentric --range 200 --drones 10   # 4. compare
```

Steps 3 and 4 for every range at once:
`run_all_tests_with_multiple_runs.py`. The individual is always the
`Parameters:` line of `<system>/ga_logs/ga_logbook_<range>.txt`. That is the
best individual of the last generation. It is preferred to the Hall of Fame,
which keeps the luckiest evaluation. A range without a logbook is an error, and
nothing runs untuned. There are two exceptions:
- centralized and naive have no logbooks yet. Their `main.py` picks the individual
  (`default_individual`), see [Centralized](#centralized) and [Naive](#naive).
- `--hand-tuned` runs the individual filled in by hand, see
  [Hand-tuned individuals](#hand-tuned-individuals).

## The scenario

| | Value | Where |
|---|---|---|
| Map | 100 x 100 cells, 20 m apart: 2000 m x 2000 m, centred on (0, 0) | `main.py` `MAP_WIDTH`, `DISTANCE_BETWEEN_CELLS` |
| Cell `i` | centre at `20 i - 1000` m, on each axis | `protocol.py` |
| Duration | 10,000 s | `SIMULATION_DURATION` |
| Drones | 3/5/7/10 in the GA, set per run elsewhere | `TRAINING_DRONE_COUNTS` |
| Flight | 50 m altitude, 15 m/s cruise, 10 m/s² max acceleration (3 m/s² before 2026-11-24) | `DRONE_SPEED`, `DRONE_MAX_ACCELERATION` |
| Communication range | 100 m by default, 100–2000 m swept | `TRANSMISSION_RANGE`, `--range` |
| Base | Gaussian around the map centre, σ = 333 m (3σ reaches the border), snapped to a cell centre, drawn again if off the map | `sample_base_position` |
| Charging capacity `Nc` | `int(drones / 3)` pads, at least 1. Hard since 2026-11-24: a drone that lands with every pad taken waits on the ground until one frees. Before, it was soft: it changed decisions, nothing forbade more | `charging_station.py`, `main.py` |
| Uncertainty growth | +0.001 per cell every second | `UNCERTAINTY_RATE`, `VANISHING_UPDATE_TIME` |
| Initial uncertainty | swarm 1.0, egocentric 0.5 (see [Known issues](#known-issues-2026-11-15)) | `INITIAL_UNCERTAINTY` |
| Wind | Random air speed N(0, 4) m/s along the heading, each drone keeps a value for 60 s (piecewise constant). Same mean drain as before, but over 100 s of cruise 0.77–1.70× of it (p1–p99, 0.86–1.25× over 1000 s) instead of 0.94–1.08×. Until 2026-11-24 a new value every second | `SHARED_BATTERY_CONFIG`, gradysim `BatteryPowerConfiguration.random_air_speed_hold` |
| Discharge assumed by the decisions | 0.00067 of the battery per second (measured cruising at 15 m/s with the wind, after the 2026-11-23 battery plugin fix; 0.00073 before) | `DISCHARGE_RATE` |
| Dead | battery ≤ 10% | `DEAD_BATTERY_THRESHOLD` |
| Recharge | 100 s on a pad, then full. On the ground, waiting or charging, the battery does not change (the battery plugin is stopped from landing to take-off, since 2026-11-24) | `TIME_TO_RECHARGE` |

The drones take off from the base. The real battery drain comes from gradysim's
`BatteryPowerPlugin`. `DISCHARGE_RATE` is only what the decisions assume. At
that rate, going from full battery to dead takes about 1,230 s of flight.

## How a drone works (`protocol.py`)

### Map and uncertainty

Each drone has its own map, `map[x, y]`, with two values per cell:
- `[..., 0]`: the uncertainty. It starts at `INITIAL_UNCERTAINTY` and grows by 0.001
  every second, on every cell, in every status (also CHARGING, WAITING and DEAD).
- `[..., 1]`: the last time the cell was seen. It is 0 before the cell has ever
  been seen.

A cell seen at time `t₀` therefore has an uncertainty of about `0.001 · (t − t₀)`.
**Its uncertainty is the time since it was last seen, scaled.**

### Camera: what a visit is

Every second a MAPPING or GOING_TO_BASE drone looks straight down from 50 m with
a 30° half angle. It sees a disk of radius `50 · tan 30° = 28.9 m`: every cell
whose centre is inside it gets uncertainty 0 and time = now. Flying at 15 m/s,
this paints a strip about 58 m wide, 2–3 cells. A drone that is CHARGING, WAITING
or DEAD does not look.

Only the camera writes a time into the map. Encounters copy the newer time of
the other drone, they never create one.

### Choosing a destination (`fitness.py`)

A drone chooses a new cell when it arrives at its destination (within the 15 m it
flies per 1 s tick), in an encounter, and, in swarm, when it hears that another
drone started charging. The candidates are the cells of a 20 x 20 window around
it (±200 m), except its own cell. For each candidate:

1. **Value of a cell:** `U/τ + 0.001`, where `U` is its uncertainty and `τ` the
   flight time to it: `τ = d/v + |v·û − v_now| / (2a)`, the distance over the cruise
   speed plus the time lost turning the current velocity `v_now` towards the cell (`û`
   the unit vector to it, `a` = `DRONE_MAX_ACCELERATION`). Straight ahead at cruise the
   second term is 0, a U-turn costs `2v/(2a)`. In a pair decision the other drone's
   velocity comes in the gossip message (`FitnessEvaluator.turn_time`, since 2026-11-24).
   It is multiplied by the **energy factor** `exp(γ · min(R, 0))`.
   `R` is the battery left after flying to the cell and from there to the base,
   minus the charge margin. A cell the drone cannot safely fly to and back from
   is damped. This damping hits the far cells first, so **as the battery drops,
   the cells near the base win**.
2. **Spread:** each cell, except those under the drone's camera right now, becomes
   a Gaussian over its neighbours (Zhang et al. 2025, Eqs. 8–11). Its width is `σ₀² (1 + α · I / max I)`, in cells². So an uncertain
   region attracts as a whole, see `variance_explained.md`.
3. **Fitness of a candidate:** the mean of that spread value over the cells the
   straight flight to it would see.

The best candidate is then compared with the **attraction of the base**,
`κ · exp(−γ · R_safe)`. Here `R_safe` = battery − energy to reach the base −
charge margin. If the base attracts more, the drone goes to the base
(GOING_TO_BASE).

**Charge margin.**
- Egocentric uses the tuned `charge_margin` as it is.
- Swarm moves it:
  `charge_margin + (b or a) · (mean battery of the others − own battery) − c · max(0, charging + 1 − Nc)`,
  never below `MIN_CHARGE_MARGIN` (0.15 for swarm since 2026-11-24, 0.20 before). `b` applies when the drone is below the mean of the others, `a`
  when it is above. With more drones charging, the margin drops, so the drone stays
  out longer.

### Encounters

1. **Heartbeat:** every second each drone broadcasts a heartbeat.
2. **Gossip reply:** a MAPPING or GOING_TO_BASE drone that hears it answers the
   sender with a gossip message: its map, position, status and battery table.
3. **Merge:** the receiver keeps, for each cell, the copy with the newer time, and
   for each drone, the battery row with the newer time.
4. **Joint decision:** if more than 10 s have passed since the last one with that
   drone, the drone with the higher id chooses the destinations of both. It adds a
   separation term to push them apart, `distance between the two cells /
   distance_between_drone_norm`, then sends the other drone its destination and
   status.

Who takes part in step 4 differs:
- **swarm:** a MAPPING or GOING_TO_BASE drone. So a drone going to the base can be
  sent back to mapping.
- **egocentric:** only when both drones are MAPPING. A MAPPING drone that meets one
  going to the base chooses again on its own, and a drone going to the base keeps going.

### Battery, recharge and death

- **Battery check:** every 5 s.
- **Arrival at the base:** the drone brakes on the approach, at
  `min(15, √(2 a d), d / 1 s)` with `a` = 1.2 m/s² and `d` the distance left. It
  is on the base once within 5 cm and below 0.1 m/s.
- **Landing:** the drone then lands and is frozen until it takes off again. Every
  routine stops: camera, heartbeat, mobility. The motors are off, so its battery
  plugin stops too: the battery does not change on the ground and no drone can die
  there. Before 2026-11-24 the plugin kept billing hover power during the charge.
- **Pads and waiting room** (since 2026-11-24, `charging_station.py`): the base has
  `Nc` pads. With a pad free and nobody waiting, the drone takes it and is CHARGING
  at once. Otherwise it is WAITING, landed in the waiting room next to the base, and
  asks for a pad every second; the pads go to the drones in the order they landed.
  So at most `Nc` drones charge at once, and the others at the base wait, landed. Before,
  any number of drones charged at once and the capacity was only a penalty of the cost,
  which let a system that ignored it get its drones back to mapping sooner.
- **Charging:** 100 s on the pad. Swarm drones broadcast a charging notification
  every second, with the time the charge started and their battery then. A WAITING
  drone sends nothing.
- **After the charge:** full battery, MAPPING, and the drone chooses a cell.
- **Death:** at ≤ 10% the drone is DEAD. It stops where it is and from then on only
  broadcasts that it is dead, every 5 s.

**Swarm battery estimate** (`estimate_drone_battery`). From the last row heard
about each drone, a swarm drone projects that drone forward. It assumes the drone:
- flies at `DISCHARGE_RATE`;
- goes to the base when it reaches `MIN_CHARGE_MARGIN` (if it was MAPPING), or after the mean
  flight time to the base (if it was GOING_TO_BASE);
- charges for 100 s and comes back full, with the same cycle repeating.

A drone in the waiting room is silent. The others keep projecting it from the last row
they heard (GOING_TO_BASE: charging from its expected arrival on) until its charging
notifications start, so a long wait shows up as a charge that seems to end too early.

This gives the mean battery of the others and how many are charging. Known-dead
drones are left out. `estimation_decision.md` measures how far off this estimate is.

### Swarm vs egocentric, in one table

| | swarm | egocentric |
|---|---|---|
| Genes | 9: the 6 below plus `a`, `b`, `c` | 6: `base_variance`, `alpha_variance_modifier`, `energy_gamma`, `charging_base_multiplier`, `distance_between_drone_norm`, `charge_margin` |
| Charge margin | moved by the swarm battery and congestion | fixed |
| Charging notification | sent and used | none |
| Re-routed in encounters while GOING_TO_BASE | yes | no |

Everything else is meant to be the same. A change to one that is not about energy
should go to the other too.

## Centralized

`centralized/` is the top of the baseline ladder of `A1.md` (naive → egocentric →
swarm → centralized). It measures how much decentralization costs. It is the swarm
drone with the information and the recharge decision made central. The full
description, the checks and the references are in
[`centralized/README.md`](centralized/README.md).

**What changes from swarm:**
- **One global map.** Every drone holds the same map array. A camera writes into it and
  everyone sees the result at once. The uncertainty grows once per second for the whole
  map.
- **The truth about the others.** Every drone reads the true battery, position and
  status of the others from a shared Python object, `world.CentralizedWorld`. It is not a
  gradysim node, so the node set is the same as swarm's.
- **No radio.** No heartbeat, gossip or notification is sent, and `main.py` builds no
  communication handler.
- **Pair decisions by distance.** Two MAPPING drones within `ENCOUNTER_RADIUS` (200 m,
  fixed) choose their cells together, with swarm's `choose_two_cells`, as an encounter
  would.
- **A central charging scheduler decides the returns.** Every second it recomputes, from
  the true state, a latest-safe slot for each drone, with at most `max(1, Nc)` drones on
  the base at once. It works backwards from the latest deadline, with a repair for drones
  whose window is narrow. A drone leaves when its slot minus its flight time has come.
  The reserve is the fixed 0.25 (`MIN_CHARGE_MARGIN`, 0.20 before 2026-11-24). A
  recharge always takes 100 s whatever the battery, so returning late is the right goal.
  The pads are hard here too: a drone sent to a full base (no free slot before its
  deadline) waits landed, and the scheduler books the drones of the waiting room at the
  moment a pad frees for them (`world.waiting_room_starts`).
- **4 genes.** The coverage genes are kept. The base attraction `κ`, the genes `a`, `b`,
  `c` and `charge_margin` are gone.

**What stays the same.** The scenario, the camera, the mobility, the battery and wind,
the seeds (run `k` has the same wind and first destinations as swarm), the cost, the GA,
the monitors, the recorder and the plots. `multiple_runs.py`, `plot_multiple_runs.py`
and `run_all_tests_with_multiple_runs.py` accept `--system centralized`. In
`plot_multiple_runs.py` it is drawn in black, dotted, with `X` markers.

**The individual.** Centralized is not tuned yet. `--range` only files the runs and
picks the individual: the coverage genes of the swarm individual tuned for that range,
or `centralized/ga_logs/ga_logbook.txt` once the GA has been run
(`python3 -m Analytical.centralized.main --mode train`). So at a given range, centralized
and swarm map with the same genes.

**Checks.** `python3 -m Analytical.centralized.check_centralized` tests:
- the scheduler against an exact MILP;
- the properties of one run: a shared map, growth once per second, no messages,
  determinism, and the same first destinations as swarm;
- a sweep of 24 seeded runs.

Every check passed on 2026-11-15.

**First results (2026-11-15).** These come from 20 paired runs per size, 200 m, 10,000 s:
centralized against the swarm runs in `swarm/multiple_runs/200/`. The table is in
`centralized/README.md`.
- **Uncertainty:** centralized's mean global uncertainty is 14.7%, 15.5%, 16.5% and 17.9%
  lower for 3, 5, 7 and 10 drones, lower in every run.
- **Deaths:** none.
- **Capacity:** centralized was over it for 83 s out of 800,000 run-seconds, never by
  more than one drone. Swarm was above `Nc` 0.7–2.5% of the time.

## Naive

`naive/` is the bottom of the baseline ladder of `A1.md`. It shows whether being
energy-aware matters at all. It is the egocentric drone with the energy logic replaced
by a fixed rule. Details are in [`naive/README.md`](naive/README.md).

**What changes from egocentric:**
- **Cells by uncertainty alone.** The energy factor of the cells (Eq. 14) is 1, so a low
  battery no longer pulls the drone towards the base.
- **A fixed return rule.** A MAPPING drone goes to the base as soon as its battery drops
  below `RETURN_BATTERY_THRESHOLD` = 0.30 (back to 0.30 on 2026-11-24 after a day at 0.25), checked every second. The base is never
  compared with the best cell, and pair decisions only send drones to map.
- **3 genes.** `base_variance`, `alpha_variance_modifier` and
  `distance_between_drone_norm`. `energy_gamma`, `charging_base_multiplier` and
  `charge_margin` are gone.
- **`INITIAL_UNCERTAINTY` = 1.0,** as in swarm and centralized. Egocentric still has 0.5.

**The individual.** Naive is not tuned yet. `naive/ga_logs/ga_logbook_<range>.txt` is used
once it exists (`all_optimizations_distance.py --system naive`). Until then it is the
coverage genes of the egocentric individual of the range. `multiple_runs.py`,
`plot_multiple_runs.py`, `all_tests_distance.py` and
`run_all_tests_with_multiple_runs.py` all take `--system naive`. In `plot_multiple_runs.py`
it is drawn in reddish purple, with a dash-dot line and hexagon markers.

**What to expect.** All the drones start full and reach 30% at about the same time, and
nothing staggers them, so they also charge together. In a test with 3 drones, all 3
were on the base at once in both runs.

## Hand-tuned individuals

Every system's `main_test.py` (swarm, egocentric, centralized, naive) has a
`HAND_TUNED_INDIVIDUAL` near the top, with one gene per line and its bounds in a comment.
It starts as `BEST_INDIVIDUAL` of `main.py`. Edit it there, then add `--hand-tuned` to
any of the scripts that otherwise read the GA logbooks:

```
python3 -m Analytical.swarm.main_test --hand-tuned --drones 5 --seed 1        # one recorded run, report + checks
python3 Analytical/all_tests_distance.py --system swarm --hand-tuned --ranges 200 --drones 5
python3 Analytical/multiple_runs.py --system swarm --hand-tuned --runs 10 --range 200 --drones 5
python3 Analytical/plot_multiple_runs.py --system swarm --hand-tuned --range 200 --drones 5
python3 Analytical/run_all_tests_with_multiple_runs.py --system swarm --hand-tuned --ranges 200 500
```

- **One individual for every range.** The same hand-tuned individual runs at every range.
  The ranges still default to those of the logbooks, and `--ranges` chooses others.
- **Separate directories.** The runs go to `test_logs_hand_tuned/` and
  `multiple_runs_hand_tuned/<range>/<drones>`, so they never mix with the runs of the
  tuned individuals. `plot_multiple_runs.py --hand-tuned` reads them from there. A
  directory refuses runs of another individual, so **after editing
  `HAND_TUNED_INDIVIDUAL`, add `--overwrite`** to `multiple_runs.py` or
  `run_all_tests_with_multiple_runs.py`.
- **Recorded in `config.json`.** It holds the individual and `"hand_tuned": true`.
- **expanding_testing.** `expanding_testing/<system>` uses the `HAND_TUNED_INDIVIDUAL` of
  `<system>/main_test.py`, as it uses that system's logbooks.
- **Conflicts.** `--hand-tuned` with `--individual` is an error.

## What is measured

Over a run, `main.py`'s `GlobalMapMonitor` samples the swarm every second
(`--sample-interval`). `multiple_runs.py` adds a `RevisitMonitor` next to it.
Neither changes the run: the same seed gives a bit-identical run with or without them.

### Global uncertainty

The **global map** is the cell-by-cell minimum over the drones' maps: each cell
as the best informed drone knows it. No drone holds it. The **global uncertainty**
is its sum over the cells. On the 100 x 100 map it starts at 10,000 (swarm) or
5,000 (egocentric).

### Drones charging and dead

- **Charging:** the number of drones in status CHARGING, compared with `Nc`. Since
  the pads (2026-11-24) it never exceeds `Nc` (from 3 drones on).
- **Waiting:** the number of drones in the waiting room (WAITING), the congestion
  since the pads. `run_<k>.pkl` stores it as `waiting`, and `summary.txt` gives each
  run's time in the waiting room, in drone·s. Older runs do not have it.
- **Dead:** the number of DEAD drones. It never goes down.

### Time between two visits of a cell

The time between two visits is how long a cell goes unseen between two visits,
from the last time a drone saw it to the next time any drone sees it again. That
is the time its uncertainty grew.

- **Last time seen:** the maximum over the drones of `map[..., 1]`. This is the
  last time any drone saw the cell, because only cameras write it.
- **Same visit or new visit:** a drone passing over a cell sees it on 2–3 ticks
  in a row, and a drone that keeps seeing it keeps the visit going. The cell is
  visited again when it is seen after more than `sample interval + 1 s` (2 s)
  unseen.
- **Accuracy:** it is read once per sample, so an interval can be up to one sample
  interval (1 s) too long. Checked against every camera write: the number of
  revisits was the same in every cell.
- **Left out:** the time before the first visit and after the last one. A cell
  seen at most once has no time between visits.
- **What is stored:** per cell, the number of revisits, their sum and their sum of
  squares (`run_<k>.pkl`, key `revisit`). That is enough for the mean and standard
  deviation of any group of cells.

### GA cost (not in `expanding_testing`)

Integrated over the run, per second:

```
cost = 0.00072 · global uncertainty
     + 1000    · dead drones
     + 30 · (exp(5 · excess / drones) − 1),   excess = max(0, charging − Nc)
```

- **Congestion since the pads (2026-11-24):** the drones CHARGING never exceed `Nc`
  (from 3 drones on), so the last term is 0. Congestion costs through the uncertainty
  instead: a drone in the waiting room does not map.
- **One individual:** the cost is averaged over 5 runs per swarm size, then summed
  over 3, 5, 7 and 10 drones.
- **Seeds:** every generation draws 5 new seeds and bases, and every individual of
  that generation uses the same ones, so they compete on the same scenarios.
- **GA settings:** 50 individuals, 20 generations, tournament of 3, SBX crossover
  (0.8), polynomial mutation (0.3), and the best individual is copied to the next
  generation.

## The figures of `plot_multiple_runs.py`

The runs are put on a common time grid of one point per second. The value at `t`
is the last one sampled at or before `t`. Several systems can be drawn on the same
figures: `--system swarm egocentric`. `expanding_testing/plot_multiple_runs_bases.py`
draws the same figures per multi-base scenario.

### Global uncertainty

The line is the mean over the runs at each time, and the band is ±1 standard
deviation over the runs. The legend gives each run's mean over the whole run, as
mean ± std over the runs.

### Drones charging above `Nc`

- **(a) When:** in 50 s windows (`--window`), the probability that more than `Nc`
  drones are charging. Each window's value is the mean over the runs.
- **(b) How much:** each run's share of time above `Nc`. One dot per run, over a
  box plot. In black, the mean with its 95% bootstrap interval.
  - The interval narrows as runs are added; the spread of the dots does not.
  - With two systems seeded alike, run `k` of one is compared with run `k` of the
    other by a paired Wilcoxon signed-rank test.
- **(c) How many at once:** the share of time with exactly `k` drones charging, on
  a log scale. Everything right of `Nc` is the congestion.

Since the pads (2026-11-24) no run with 3 drones or more goes above `Nc`, so these
panels only show something on older runs. The congestion is in the `waiting` series.

Bootstrap: 5000 resamples, seeded, so the figure is the same on every call.

### Time between two visits of a cell, by distance to the base

The **x axis** is the distance from the centre of a cell to its nearest base,
which is the only base in `swarm/` and `egocentric/`. The **line** is the mean
time between two visits of the cells at that distance, over every revisit of
every run. The **band** is ±1 standard deviation of those same revisits.

**The bins (`--distance-bin`, 100 m by default).** The cells are grouped by
distance, and each group gives one point:

- **The bins:** 0–100 m, 100–200 m, 200–300 m, and so on, with no gaps. A cell
  goes to bin `floor(distance / 100)`, so a cell exactly on an edge goes into the
  bin above. For example, the cell 3 across and 4 up from the base is at
  `20 · √(3² + 4²) = 100 m`, so it is in 100–200 m.
- **Where the points are drawn:** at the centre of each bin, at 50, 150, 250 m and
  so on. That is why the line starts at 50 m.
- **The first point, 0–100 m:** about 69 cells per run, from the base cell (0 m)
  through 20, 28 (20√2), 40, 45 m and so on, up to just under 100 m. Its time is
  short, because drones take off and land there, and because when their battery is
  low the energy factor sends them to map near the base.

**Why bins are needed.** The distance takes only fixed values here. The cells are
on a 20 m grid and the bases are snapped to cell centres, so every distance is
`20 · √(i² + j²)` m. Each exact distance has very few cells. In 6 test runs (5
drones, 3000 s) there were:
- 3,154 different distances;
- a median of 6 cells and 12 revisits per distance;
- fewer than 10 revisits at 46% of the distances.

A mean of 12 revisits that can each be anywhere from seconds to over 1,000 s is
mostly noise. Grouping nearby distances fixes that. Any smooth curve needs such a
width, and a moving average would only call it something else.

**Choosing the width:**
- Narrower bins show more detail and more noise.
- 20 m (one cell) is the finest width that means anything.
- The camera strip is about 58 m wide, so detail finer than that is not really
  in the data.
- With 20 runs of 10,000 s, 40–50 m should still be smooth.

**Reading it:**
- **Far distances are not drawn everywhere:** the bases move between runs, so the
  far distances exist only in the runs with a base near a corner. A bin is drawn
  only where at least half of the runs had a revisit in it (`MIN_RUN_SHARE`).
- **The band is wide:** it shows how much single revisits differ, not how much
  the runs differ. Revisit times are spread about as widely as their mean, so the
  band is about as wide as the mean and is cut off at 0. It does not narrow as runs
  are added. This is not the same as the band of the uncertainty figure.
- **The mean can read too low:** cells seen at most once are left out. The script
  prints their share. Where many cells are rarely seen (for example when drones
  die), the line is lower than the truth, because the longest gaps are the ones
  missing.
- **The legend** gives the mean ± std of every revisit over the whole map.

## Seeds and fair comparisons

- **What a seed fixes:** the seed of a run fixes each drone's first destination
  and its wind. Each drone has its own random streams, built from `(seed, drone id)`,
  so drawing for one drone never moves another, and the scenario stays the same
  even after two individuals' decisions diverge.
- **Run `k`:** in `multiple_runs.py`, run `k` uses seed `--seed + k`, and its
  bases are drawn from that seed. So run `k` has the same base and wind in swarm
  and in egocentric, and on every range.
- **Comparing over many runs, not one:** the same seed does **not** reproduce a
  whole run once the set of nodes changes. For example, adding the base nodes of
  `expanding_testing` makes runs diverge within about 4 s. gradysim orders the
  events of the same instant by their position in its event queue. Compare over
  many runs, not one.

## Files the scripts write

| File | Content |
|---|---|
| `ga_logbook_<range>.txt` | Per generation: avg/std/min/max cost and the terms of the best (`min_uncertainty`, `min_dead`, `min_charging`). At the end, the best individual of the last generation and its `Parameters:` line |
| `ga_output_<range>.txt` | Everything the GA printed, including each generation's seeds and bases |
| `test_logs/.../test_output.txt` | The report of `main_test.py`: per drone results, the status episodes, the battery estimate error, the pads and waiting room of the base, and the checks |
| `test_logs/.../test_recording.pkl` | The full recording: every drone's map and state every 10 s, read by `<system>/plots/` |
| `test_logs/.../main_test.log` | The protocol log, every routine of every drone |
| `multiple_runs/.../run_<k>.pkl` | One run: the uncertainty, charging, waiting (since 2026-11-24) and dead series, the base(s), the seed, the cost and its terms (not in `expanding_testing`), each drone's final state, and the revisit sums per cell |
| `multiple_runs/.../config.json` | The scenario of the directory. A directory refuses runs of another scenario |
| `multiple_runs/.../summary.txt` | One line per run and the mean ± std over the runs |
| `multiple_runs/ranges_summary.txt` | `run_all_tests_with_multiple_runs.py`: the same per range |

`main_test.py` exits with status 1 when a check fails. The checks are:
- never more drones charging at once than the pads of the base;
- every drone mapped something;
- the global map is at least as good as the best drone's;
- every drone recharged at least once;
- no drone died;
- no drone flew farther from the map centre than 1.2 times half the map side.

`multiple_runs.py` skips the runs that already exist, so `--runs 20` after
`--runs 10` only runs the new 10. `--overwrite` runs them all again.

## Known issues (2026-11-15)

- **Runs before 2026-11-24 are not comparable with newer ones.** That day the wind
  became piecewise constant (held 60 s), the base got hard pads with a waiting room,
  and the battery stopped draining on the ground.
- **Revisit data in old runs:** run files written before 2026-11-15 have no
  revisit data, and neither do the MARL runs. The revisit figure leaves them out
  and says so. Run them again with `--overwrite`, or into a new directory, to get it.
- **Initial uncertainty:** `INITIAL_UNCERTAINTY` is 1.0 in swarm and 0.5 in
  egocentric, here and in `expanding_testing`. It is not about energy, so the two
  systems start the global uncertainty at different values (10,000 against 5,000).
- **The egocentric 200 m individual:** it goes back to the base too late. In 6
  test runs of 3000 s with 5 drones, all 5 drones died in every run, though some
  had recharged before. `BEST_INDIVIDUAL` in `egocentric/main.py` is still the one
  from the old 50 x 50 scenario.
- **Absolute paths:** an absolute `--logs-directory` or `--logs-root` makes every
  system read and write the same directory, because the path is joined to each
  system's directory.

## Other documents

- [`expanding_testing/README.md`](expanding_testing/README.md): several bases,
  nearest-base rules, the multi-base scenarios.
- [`centralized/README.md`](centralized/README.md): the centralized reference, its
  scheduler, the checks and the references.
- [`naive/README.md`](naive/README.md): the naive baseline.
- [`waiting_room.md`](waiting_room.md): the hard base capacity (pads and waiting room) of
  2026-11-24, what changed and where in the code.
- [`estimation_decision.md`](estimation_decision.md): the battery estimate against
  the real battery.
- [`variance_explained.md`](variance_explained.md): the Gaussian width genes.
- [`../MARL/documentation.md`](../MARL/documentation.md): the learning version of the
  same protocol.
