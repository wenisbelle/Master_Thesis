# Centralized: the swarm with perfect information and a central charging scheduler

`centralized/` is the top rung of the baseline ladder in [`A1.md`](../../../A1.md):
naive → egocentric → swarm → **centralized**. It answers the question "how much does
decentralization cost?". It runs the same drones as `swarm/` on the same scenario.
The only differences are what each drone knows and who decides when it recharges:

- **Information.** Every drone reads one global map and the true battery, position and
  status of every other drone. Nothing is estimated and nothing is sent.
- **Recharging.** A central scheduler decides who goes to the base and when, so that at
  most `Nc` drones charge at once and each drone returns as late as is safe.

It is a **reference, not a proven upper bound**. The drones still choose where to map
with the greedy fitness of swarm, so a better coverage planner could beat it. What it
bounds is the part the A1 claim is about: energy coordination and congestion with full
information.

The code as of 2026-11-15 is the source of truth. Where this file and the code disagree,
the code is right.

## Contents

| File | What it is |
|---|---|
| `world.py` | **New.** `CentralizedWorld`: the global map, the registry of drones and the charging scheduler (`plan_charging`, `place_slots`, `latest_slot`) |
| `protocol.py` | The swarm drone without a radio, see [the differences](#what-differs-from-swarm) |
| `fitness.py` | swarm's `fitness.py` without the base attraction and without the genes `a`, `b`, `c` |
| `main.py` | swarm's scenario, cost, monitors and GA, with 4 genes and no communication handler |
| `main_test.py` | swarm's single recorded test, with a charging section and one extra check in the report |
| `check_centralized.py` | **New.** The checks of this system, see [Checks](#checks) |
| `recorder.py`, `estimation_metrics.py`, `visualization.py`, `plots/` | Copied from swarm unchanged (they are the same in swarm and egocentric) |

## What differs from swarm

| | swarm | centralized |
|---|---|---|
| Map | one per drone, merged in encounters | **one global map**: every drone's `map` is the same array (`world.map`) |
| Uncertainty growth | each drone's map, every second | the global map, once per second (`world.grow_uncertainty`) |
| Battery of the others | estimated from gossip (`estimate_drone_battery`) | **true**, read from the world |
| Messages | heartbeat, gossip, goto, dead and charging notifications | **none**. `main.py` builds no communication handler |
| Encounter (pair decision) | heard heartbeat within the radio range, higher id decides both | two MAPPING drones closer than `ENCOUNTER_RADIUS` (100 m, fixed), higher id decides both, at most once per pair every 10 s. The same `choose_two_cells` with the separation term |
| Who takes part in a pair decision | MAPPING or GOING_TO_BASE | MAPPING only: a drone sent to the base keeps going |
| Going to the base | the drone compares the base attraction `κ·exp(−γ·R_safe)` with its best cell | **the scheduler** sends it, checked on every 1 s mobility tick |
| Reserve kept for the trip home | `charge_margin` + swarm and congestion terms, at least `MIN_CHARGE_MARGIN` = 0.15 | fixed `MIN_CHARGE_MARGIN` = 0.25 (0.20 before 2026-11-24) |
| Energy factor of the cells (Eq. 14) | uses the tuned `charge_margin` | uses 0.25, the same reserve as the scheduler |
| Genes | 9 | **4**: `base_variance`, `alpha_variance_modifier`, `energy_gamma`, `distance_between_drone_norm` |

Everything else is swarm's code, copied with its comments: the camera, the mobility and
the braking approach to the base, the battery plugin and the wind, the 300 s recharge,
death at 10%, the take-off from the base, the random streams from `(seed, drone id)`,
`INITIAL_UNCERTAINTY` = 1.0, the cost, the GA and its bounds, the monitors.

### Kept so the recorder and the plots read it like swarm

`swarm_battery_status`, `aproximate_battery_consumption_model` and
`estimate_drone_battery` keep their names and signatures but return the truth.
`fitness.charge_margin` returns the fixed reserve. The battery-estimation part of the
report and of `plots/plot_estimation.py` therefore shows an error of exactly 0, which is
what perfect information means. The decisions do not use these functions.

Two changes to the per-drone results are needed because the map is shared:
- `unvisited_cells` is the swarm's, the same for every drone.
- A new `own_cells_seen` counts the cells this drone's own camera saw. The per-drone
  coverage of the report and the check "every drone mapped something" use it.

## The charging scheduler (`world.py`)

A full recharge always takes `TIME_TO_RECHARGE` (300 s since 2026-11-25, 100 s before) and
fills the battery, whatever was left. A drone
that recharges early therefore only wastes battery. The best policy is to return **as
late as is safe**, staggered so that no more than `Nc` drones are on the base at once.
The base is busy: a drone charges for 300 s after about every 1,100–1,300 s of flight,
against `Nc = int(n/3)`, so with n a multiple of 3 each pad is in use about 60% of the
time (about 24% with the 100 s charge before 2026-11-25). With other n it is more: 5
drones share a single pad, about 100%. Congestion also comes from the drones being in
phase, because they all take off full at t = 0.

The scheduler is computed again from the true state every time a MAPPING drone asks, once
per mobility tick (1 s):

1. **Booked charges.** These are the drones CHARGING, from the start of their charge, the
   drones GOING_TO_BASE, at the slot they left for, and (since 2026-11-24) the drones in
   the waiting room of the base, at the moment a pad frees for them
   (`waiting_room_starts`: in the order they landed, each at the end of the earliest
   charge still on a pad).
2. **Deadlines.** Each MAPPING drone gets its latest safe arrival,
   `now + (battery − reserve) / (1.15 · DISCHARGE_RATE)`. Arriving then still leaves the
   reserve, `MIN_CHARGE_MARGIN` = 0.25 (0.20 before 2026-11-24).
   The flight time to the base cancels out of this formula.
3. **Slots.** The drones are taken from the latest deadline to the earliest. Each gets
   the latest charging slot that starts no later than its deadline, during which fewer
   than `max(1, Nc)` other charges are on the base. A slot is `TIME_TO_RECHARGE` (300 s) + `SLOT_GUARD` (15 s)
   long. This is backward earliest-deadline-first.
4. **Repair.** A drone with a narrow window (near its deadline and far from the base) can
   find every slot it can reach taken by a drone placed earlier that had room to go
   earlier. First a swap is tried: the drone takes that slot and the other drone is
   placed again. If that fails, the drone is moved to the front of the order and the pass
   is made again.
5. **No free slot.** The drone gets the start before its deadline that adds the fewest
   seconds above capacity. A drone is never sent late to avoid congestion.
6. **Departure.** The drone leaves when `slot − flight time` has come. Because it only
   checks once per tick, it is always a little late for its slot. A free slot missed by at
   most `SLOT_GUARD` means "leave now".

The flight time is `distance / 15 + 15 / (2 · 0.4a) + APPROACH_EXTRA_TIME` (0 since 2026-11-24, 7 s before). The first two terms are swarm's
`EXPECTED_TIME_TO_BASE` formula (cruise plus braking).

**The pads are hard (since 2026-11-24).** As in every system, the base has `max(1, Nc)`
pads (the base node, `centralized/recharge_base.py`). A drone that lands with every pad taken waits on the
ground, with its battery frozen, until one frees. So what used to be an overrun of the
capacity (step 5, or a drain above the planning factor) is now a short wait, and it can
no longer exceed the capacity or kill a drone on the base.

### Where the constants come from

Each one fixes a failure the sweep found (2026-11-15):

| Constant | Value | Measured | What went wrong without it |
|---|---|---|---|
| `APPROACH_EXTRA_TIME` | 0 s (7 s until 2026-11-24, measured at 3 m/s²; not re-measured at 10 m/s²) | Over 270 trips at 3 m/s², the real flight to the base took 7.3 ± 2.9 s longer than `d/v + v/2a`. The extra time is the turn towards the base and the last centimetres of the approach | Charges planned back to back overlapped by a few seconds |
| `PLANNING_DRAIN_FACTOR` | 1.15 | Over 100 s of MAPPING, the real drain is 0.999 of `DISCHARGE_RATE` on average, 1.08 at the 95th percentile, 1.12 at the 99th and 1.22 at most | With the nominal rate, a drone that drained faster saw its deadline move earlier, after the drones planned around it had left. Up to 5 drones charged at once with `Nc = 3`. With 1.15 a deadline only moves later. The cost is about 0.01 of battery at arrival |
| Booking of a drone GOING_TO_BASE | its slot | While a drone turns towards the base its distance barely drops, so `now + flight time` slid later by up to 1 s per second | On a base full to capacity, a 2 s slide pushed another drone 105 s earlier, and a third was left with no slot |
| `SLOT_GUARD` | 15 s | Absorbs the remaining error in the arrival time (±3 s) and one tick of delay | |

A side note on swarm: its `EXPECTED_TIME_TO_BASE` uses the same `d/v + v/2a` formula, so
it is probably about 7 s short too. That is not fixed there.

### Why not an exact optimizer

The problem of placing the slots is that of Simons (1983): jobs of equal length (the
charges), identical machines (the capacity), release times (the earliest arrival) and
deadlines. A time-indexed MILP solves one instance exactly in about a second, but the
plan is needed on every tick of every drone. `check_centralized.py` compares the greedy
with that MILP instead (see [Checks](#checks)). Planning routes and charges together
(Kim, Song & Morrison 2013; Mitchell et al. 2015) does not scale to 10,000 cells and
10 drones. It would also replace the coverage planner, so the gap to swarm would no
longer measure only the energy coordination. See [Related work](#related-work) for how
each paper relates to this scheduler.

## The individual

Centralized has not been tuned yet. Until it is, it runs with **the coverage genes of the
swarm individual** (base_variance, alpha, gamma, drone_norm, picked by name):

- `BEST_INDIVIDUAL` in `main.py` comes from the swarm individual tuned at 200 m
  (`swarm/ga_logs/ga_logbook_200.txt`).
- `multiple_runs.py --system centralized --range R` uses `default_individual(R)`. That is
  `centralized/ga_logs/ga_logbook.txt` when it exists, otherwise the coverage genes of
  `swarm/ga_logs/ga_logbook_<R>.txt`. The origin is printed and saved in `config.json`.

So at every range, centralized and swarm map with the same genes and differ only in
information and in the energy decision. That makes it a controlled comparison.

A1.md asks for a fair retune with the same GA budget. The GA here is swarm's with 4 genes,
and it does not depend on the range because there is no radio:

```
python3 -m Analytical.centralized.main --mode train --logbook Analytical/centralized/ga_logs/ga_logbook.txt
```

Once that logbook exists, it becomes the default individual of every range.

## Running

From `src/`, with gradysim on the path (see [`../README.md`](../README.md#running)):

```
python3 -m Analytical.centralized.main_test --drones 5 --seed 1             # one recorded run, report + checks
python3 -m Analytical.centralized.main_test --base-position 400 -200 --encounter-radius 500
python3 -m Analytical.centralized.main_test --hand-tuned                     # HAND_TUNED_INDIVIDUAL of main_test.py
python3 Analytical/multiple_runs.py --system centralized --runs 20 --range 200 --drones 10
python3 Analytical/plot_multiple_runs.py --system swarm centralized --range 200 --drones 10
python3 Analytical/run_all_tests_with_multiple_runs.py --system centralized   # every swarm range
python3 -m Analytical.centralized.check_centralized                        # the checks below
```

- `main_test.py` has no `--transmission-range`, because there is no radio.
- `--encounter-radius` sets the pair-decision distance.
- The plots of one run work from `centralized/plots/` as in swarm, for example
  `python3 plot_drones.py --directory logs`.
- In `plot_multiple_runs.py`, centralized is drawn in black, dotted, with `X` markers.

## Same metrics as swarm and egocentric

The monitors (`GlobalMapMonitor`, `RevisitMonitor`), the cost and its terms, the run
files of `multiple_runs.py` and the recording of `main_test.py` are the same as in swarm.
Every script that reads them reads centralized too. The global map that the monitors
assemble (the minimum over the drones' maps) is the shared map itself.

The report of `main_test.py` adds two things:
- A **Charging (the scheduler)** section: departures, the battery at departure and at
  arrival, how far each slot was moved earlier for the capacity, and the most drones
  charging at once.
- A **Base (pads and waiting room)** section, as in every system: landings, how many
  found every pad taken, and how long they waited.
- A check that no more than `max(1, Nc)` drones were ever charging at once, the pads of
  the base. Since the pads it holds by construction.

## Checks

`check_centralized.py` runs three groups of checks and exits 1 if any fails.

**scheduler.** `place_slots` on random small cases:
- 5,000 float cases: no slot is after its deadline, and no slot is before the drone can
  arrive.
- Integer cases against an exact time-indexed MILP (scipy's HiGHS), on 500 cases (seed
  7). 390 had a schedule within capacity; 110 had none.
  - The greedy alone found one in 382 of the 390, always as late as the optimum.
  - The 8 misses were all drones with a narrow window. This is why the repair exists.
- **Greedy with the repair:** found a schedule in 389 of the 390, always as late as the
  optimum. The one miss needs two drones moved at once (one 240 s earlier), which neither
  the swap nor the reordering does. In that case the fallback keeps the extra time on the
  base as short as possible.
- With the default 150 cases (seed 0), it found a schedule in 123 of 123 feasible cases,
  all optimal.

**run.** One seeded run (5 drones, 2,500 s, seed 1; also 7 drones, 3,000 s, seed 3 with the base at (680, −860)):
- every drone holds the world's map arrays on every sample;
- the map grows once per second, not once per drone: every cell's uncertainty matches
  the rate times the time since it was last seen, within one growth step;
- there is no communication handler, and nothing is sent or received;
- the same seed gives the same run, and a recorded run (`main_test.py`) matches an
  unrecorded one;
- the same seed gives every drone the same first destination as swarm, so run `k` is the
  same scenario in both systems.

**sweep.** 24 runs of 4,000 s: 3, 5, 7 and 10 drones × seeds 0–2 × the base at the centre
or drawn as in `multiple_runs.py`. All of these hold:
- no drone died;
- never more than `max(1, Nc)` drones were charging at once;
- every drone recharged, at least 3 times in every run;
- the lowest battery at arrival was 0.191, against a 0.20 reserve;
- no drone left the map.

The scheduler moved 33–67% of the departures earlier to make room on the base.

All 7 plot scripts in `plots/` ran on a centralized recording.

## Results so far (2026-11-15)

**Setup.** `multiple_runs.py --system centralized --range 200 --runs 20`, for 3, 5, 7 and
10 drones, 10,000 s each, in `centralized/multiple_runs/200/<drones>`.
- **Individual:** the coverage genes of the swarm individual tuned at 200 m.
- **Swarm runs:** those already in `swarm/multiple_runs/200/<drones>` (2026-11-03). Swarm's
  code has not changed since, apart from `BEST_INDIVIDUAL`, which `multiple_runs.py` does
  not use.
- **Paired:** run `k` has the same seed, wind, first destinations and base in both systems.
- **Same pair-decision radius:** these runs used `ENCOUNTER_RADIUS` = 200 m (its value until 2026-11-16), equal to swarm's radio range,
  so the two differ only in information and in the energy decision.

| Drones | Mean global uncertainty, centralized | swarm | Paired difference | Time above Nc, centralized | swarm | Most charging at once, centralized | swarm | Cost, centralized | swarm |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 15,251 ± 509 | 17,899 ± 714 | −14.7% ± 3.1 | 0.00% | 0.67% | 1.00 | 1.75 | 109,829 | 137,553 |
| 5 | 9,896 ± 458 | 11,729 ± 700 | −15.5% ± 3.0 | 0.00% | 2.50% | 1.05 | 3.35 | 71,269 | 111,231 |
| 7 | 7,289 ± 430 | 8,741 ± 646 | −16.5% ± 2.9 | 0.00% | 1.01% | 2.05 | 3.75 | 52,493 | 68,605 |
| 10 | 5,223 ± 446 | 6,372 ± 608 | −17.9% ± 3.7 | 0.04% | 0.81% | 3.10 | 4.75 | 37,687 | 48,331 |

The ± values are the standard deviation over the 20 runs. In "Most charging at once"
and "Cost" each value is the mean over the runs.

- **Uncertainty.** Centralized is lower in **20 of 20 runs at every size** (paired
  Wilcoxon p = 2·10⁻⁶ each). So at 200 m, decentralization costs swarm 15–18% of mean
  global uncertainty.
- **No deaths.** No drone died in either system.
- **Capacity.** In the 80 centralized runs, more than `max(1, Nc)` drones were charging
  for **83 s out of 800,000** run-seconds, in 4 runs, never more than one drone over (see
  Limitations). Swarm was above `Nc` 0.7–2.5% of the time, with up to 3–4 drones over.
- **Time between two visits of a cell.** Centralized: 1,754, 1,157, 860 and 621 s for 3, 5,
  7 and 10 drones. The swarm runs are older than `RevisitMonitor` and have no revisit data.

This does not yet say where the gap comes from: the shared map, the true batteries, or
the scheduler. The "oracle info" condition of `estimation_decision.md` would separate
them.

## Limitations

- **Not a proven bound.** The coverage is still swarm's greedy fitness, and the scheduler
  is a heuristic tested against an exact solver on small cases, not proven optimal.
- **Rare overruns of the capacity (before 2026-11-24).** In 4 of the 80 full runs, one
  drone more than `max(1, Nc)` charged for 1–67 s. Since the pads, such a drone waits
  landed instead (see the scheduler).
  - **The 67 s case** (10 drones, seed 12, t ≈ 2,018 s): two drones left on the same tick
    and a third half a second later. Four arrivals were then bunched in a window with room
    for three. In that second cycle, the slots were packed against the capacity, and the
    crunch only became visible once no drone could move early enough.
  - **The 1 s case** (5 drones, seed 8): a drain more than 15% above `DISCHARGE_RATE` for
    about two minutes.
  - **Effect on the checks:** the check of `main_test.py` and of the sweep fails on such a
    run. That is on purpose, so it stays visible.
  - **What might remove them:** planning further ahead, or a longer `SLOT_GUARD`.
- **Fixed pair-decision radius.** `ENCOUNTER_RADIUS` is 100 m, the default range of swarm (200 m before 2026-11-16),
  and does not follow `--range`. This keeps centralized the same at every range, so it is
  a flat reference line. At 2,000 m swarm makes many more pair decisions than centralized.
- **Not tuned yet.** The coverage genes come from swarm, see [The individual](#the-individual).
- **Inherited artefacts (fixed 2026-11-24).** Until then the battery drained at hover while
  CHARGING, as in swarm (`estimation_decision.md`). Since the pads, the battery plugin is
  stopped from landing to take-off, in every system.
- **A wave of early first returns.** At t = 0 every drone is full, so the first cycle has
  to be staggered: some drones return 200–450 s early once. Later cycles stay staggered.

## Related work

The scheduler is not an implementation of any one paper. It is a greedy written for this
problem: backward earliest-deadline-first with a repair (see
[The charging scheduler](#the-charging-scheduler-worldpy)). Each paper below gives it one
piece:
- Simons (1983) defines the problem it solves.
- Mitchell et al. (2015) and Kim, Song & Morrison (2013) are the full centralized methods
  it simplifies.
- Bethke, How & Vian (2008) motivates one design choice.
- Hartuv et al. (2018) and Derenick et al. (2011) are related work only.

What is said here about each paper comes from its abstract (see [References](#references)).

| Paper | What this method takes from it | What it leaves out |
|---|---|---|
| Simons (1983) | The formal problem that placing the slots solves | Simons' algorithm. A greedy, checked against an exact MILP, is used instead |
| Mitchell et al. (2015) | The setting: one central planner decides when fuel-limited robots refuel during persistent coverage | Planning the visits. The drones still choose where to map |
| Kim, Song & Morrison (2013) | The exact joint formulation (a MILP) that this method approximates | The MILP. It is far too large to solve on every tick |
| Bethke, How & Vian (2008) | Fuel use is uncertain, so plan conservatively and plan again often | Their planning method |
| Hartuv, Agmon & Kraus (2018) | The same goal: keep a persistent task going under battery limits | Spare drones. There are none here |
| Derenick, Michael & Kumar (2011) | Nothing in this system. It is the decentralized counterpart, closer to swarm | Everything |

### Simons (1983): the problem the slots solve

Simons schedules jobs of equal length on identical parallel machines. Each job has a release
time, before which it cannot start, and a deadline. The paper gives a polynomial-time
algorithm for this problem. Placing the charging slots ([`place_slots`](world.py)) is an
instance of it:

| Simons | This scheduler |
|---|---|
| A job, all of equal length | One charge: `TIME_TO_RECHARGE` + `SLOT_GUARD` = 315 s |
| The identical machines | The `max(1, Nc)` places on the base |
| Release time | The earliest arrival: `now` + the flight time to the base |
| Deadline | The latest safe arrival, `now + (battery − 0.20) / (1.15 · DISCHARGE_RATE)`. This bounds the start of the charge, so Simons' deadline (on the end of the job) is this plus 315 s |
| (no equivalent) | Booked charges: the drones CHARGING or GOING_TO_BASE. They are fixed, and act as machines that are busy for a while |

How it is used:
- **To name the problem.** In the thesis the scheduler can be described as a greedy
  heuristic for Simons' problem, solved again on every tick (a rolling horizon).
- **The objective differs.** Simons asks for a schedule that meets every deadline. This
  scheduler also wants every charge as late as possible, because a charge that starts early
  only wastes battery. So it does not run Simons' algorithm. It runs the backward greedy,
  which places each charge as late as its deadline and the capacity allow.
- **The exact check.** The time-indexed MILP in `check_centralized.py` solves this same
  problem with integer times. The greedy with its repair matched the optimum in 389 of
  390 feasible cases (see [Checks](#checks)).

### Mitchell et al. (2015): the setting

Mitchell et al. study fuel-limited robots that cover an area persistently. One central
planner decides both the visits and the refuelling. It is the closest published
counterpart of this system: persistent coverage, limited batteries, refuelling at a base,
and all the information in one place.

- **Taken.** The refuelling is decided centrally from the state of the whole team.
- **Left out.** Planning the visits. Here the drones still choose where to map with swarm's
  greedy fitness and the same coverage genes. If centralized also planned the routes, the
  gap to swarm would mix a better coverage planner with better energy coordination. The A1
  claim is only about the second.
- **The cost.** Centralized is not an upper bound (see [Limitations](#limitations)).
- **In the thesis.** Cite it as the full centralized approach. This reference is
  deliberately only its refuelling half.

### Kim, Song & Morrison (2013): the exact version

Kim, Song & Morrison schedule UAVs and the fuel service stations they share, for long
missions, with a mixed-integer linear program (MILP). It is the exact joint problem:
which UAV does which task, and when and where each one refuels. As here, the stations
are shared, so one UAV's refuelling constrains the others.

- **Taken.** The base is a shared resource that the charges have to be scheduled on.
- **Left out.** The MILP. A joint MILP over 10,000 cells and 10 drones, solved again every
  second, is far too large. Even the MILP over the slots alone takes about a second per
  instance, which is why it is only used as a check. See
  [Why not an exact optimizer](#why-not-an-exact-optimizer).
- **In the thesis.** Cite it, with Mitchell et al., as the exact methods this scheduler
  approximates.

### Bethke, How & Vian (2008): fuel use is uncertain

Bethke, How & Vian plan persistent surveillance with UAV teams whose fuel use is uncertain,
so the plan has to allow for a UAV burning more fuel than expected. Their method is not
used here, but two choices follow the same idea:

- **A rolling horizon.** The plan is computed again from the true batteries on every 1 s
  tick. A drone that drains faster gets an earlier deadline at once.
- **A conservative drain.** Deadlines are computed with `PLANNING_DRAIN_FACTOR` = 1.15
  times `DISCHARGE_RATE`. This way a deadline only moves later. It never moves earlier
  after the other drones have been planned around it. The value 1.15 was measured in this
  simulator (99th percentile 1.12, see
  [Where the constants come from](#where-the-constants-come-from)). It is not from the
  paper.

### Hartuv, Agmon & Kraus (2018): same goal, other means

Hartuv, Agmon & Kraus schedule spare drones that take over from drones whose battery runs
low, so that a persistent task never stops. Their goal is the same as this scheduler's:
keep the task going while drones leave to recharge.

- **The difference.** This system has no spares. A drone that charges leaves the map for
  300 s and the others keep mapping. The scheduler only staggers the charges so the base
  is never over capacity, and keeps each drone in the air as long as is safe.
- **Not used in the code.** It is related work only.

### Derenick, Michael & Kumar (2011): the decentralized counterpart

Derenick, Michael & Kumar study coverage control in which each robot, with no central
planner, balances covering its area against going to dock and recharge.

- **Closer to swarm.** Its drone also weighs the pull of the base against its best cell.
  Nothing in centralized comes from this paper.
- **In the thesis.** Cite it in the related work on decentralized energy-aware coverage,
  as the contrast to the central scheduler here.

## References

These were checked against their catalogue records and abstracts, not their full texts.

- B. Simons, "Multiprocessor scheduling of unit-time jobs with arbitrary release times and
  deadlines", *SIAM J. Comput.* 12(2):294–299, 1983. doi:10.1137/0212018. The slot
  problem: equal-length jobs, identical machines, release times and deadlines.
- D. Mitchell, M. Corah, N. Chakraborty, K. Sycara, N. Michael, "Multi-robot long-term
  persistent coverage with fuel constrained robots", *ICRA* 2015, pp. 1093–1099.
  doi:10.1109/ICRA.2015.7139312. A centralized schedule of visits and refuelling.
- J. Kim, B. D. Song, J. R. Morrison, "On the scheduling of systems of UAVs and fuel
  service stations for long-term mission fulfillment", *J. Intell. Robot. Syst.*
  70:347–359, 2013. doi:10.1007/s10846-012-9727-0. A MILP that schedules UAVs and shared
  bases; the exact version, too large here.
- E. Hartuv, N. Agmon, S. Kraus, "Scheduling spare drones for persistent task performance
  under energy constraints", *AAMAS* 2018. Replacement schedules under battery limits.
  The page numbers differ between sources.
- B. Bethke, J. P. How, J. Vian, "Group health management of UAV teams with applications
  to persistent surveillance", *ACC* 2008, pp. 3145–3150. doi:10.1109/ACC.2008.4586976.
  Planning when fuel use is uncertain, the reason the plan is recomputed every tick.
- J. Derenick, N. Michael, V. Kumar, "Energy-aware coverage control with docking for robot
  teams", *IROS* 2011, pp. 3667–3672. The decentralized counterpart, for related work.
