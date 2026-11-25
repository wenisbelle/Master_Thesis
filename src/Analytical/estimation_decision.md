# Battery estimate vs real battery, and the decision to return to the base

Status (2026-11-14): Step 1 is implemented and piloted. Steps 2 and 3 have not started.
Goal: support the main claim of [A1.md](../../A1.md) with numbers on how good the gossip
estimate is, and how much its error changes the results.

## The question

It has two parts. A reviewer will want both answered:

1. **How wrong is the estimate?** What each drone believes about the others' batteries,
   compared with their real batteries.
2. **How much does that error change the results?** Coverage, dead drones, congestion.

Answering only the first doesn't show that the error matters. Answering only the second
doesn't explain why it matters.

## How the estimate reaches the decision (swarm only)

1. **Belief table.** Each drone keeps `swarm_battery_status`, an (N, 3) array with one row
   per drone: `[battery, observation time, status]`. Gossip at encounters and the
   charging / dead notifications update it. The newest observation time wins.
2. **Projection to now.** `estimate_drone_battery` ([protocol.py:416](swarm/protocol.py#L416))
   moves a row forward from its observation time to the current time:
   - the drone discharges linearly at `DISCHARGE_RATE` (0.00067 per s; 0.00073 before the 2026-11-23 battery plugin fix);
   - a drone seen MAPPING is assumed to fly until its battery reaches
     `MIN_CHARGE_MARGIN` (0.20), then go to the base;
   - a drone seen GOING_TO_BASE reaches the base `EXPECTED_TIME_TO_BASE` after the observation;
   - at the base it charges for `TIME_TO_RECHARGE` (300 s since 2026-11-25, 100 s before) with its battery held constant
     (true since 2026-11-24; a wait for a pad, see the base node in `swarm/recharge_base.py`, is not modelled),
     then leaves with a full battery and the cycle repeats.
3. **Swarm terms.** `aproximate_battery_consumption_model`
   ([protocol.py:375](swarm/protocol.py#L375)) reduces the projected rows of the other drones
   not known to be DEAD to two numbers:
   - **m̂**: their mean battery;
   - **N̂**: how many are charging.

   With nobody left to compare with, it returns the drone's own battery and 0.
4. **Margin.** `charge_margin` ([fitness.py:684](swarm/fitness.py#L684)):

   ```
   delta  = m̂ - own battery
   margin = max(MIN_CHARGE_MARGIN,
                CHARGE_MARGIN + (b if delta > 0 else a) * delta - c * max(0, N̂ + 1 - Nc))
   ```

5. **Base fitness.** `recharge_base_fitness` ([fitness.py:702](swarm/fitness.py#L702)):
   `R_safe = battery - (distance / speed) * DISCHARGE_RATE - margin`, and the base scores
   `K * exp(-GAMMA * R_safe)`. The drone goes to the base when this beats the map cells.

Egocentric has no step 1 to 4: its margin is the tuned `CHARGE_MARGIN` alone.

### Why the error matters

The genes amplify it. With the default individual in [main.py](swarm/main.py) (tuned at
200 m: a = 3.42, b = 8.16, c = 4.82, `CHARGE_MARGIN` 0.49):
- an error of 0.05 in m̂ moves the margin by up to 0.41 (8.16 × 0.05), and the margin
  only goes from 0.20 to 0.5;
- one drone wrongly counted as charging moves the margin by −4.82, straight to the 0.20 floor.

### Where the error comes from

- **Age of the information.** Out of range and with no encounter, a row only gets older.
- **Model mismatch.** The projection sends MAPPING drones home at 0.20. Real drones decide at
  about 0.45–0.58, so a recharge the observer did not hear about is missed: it believes the
  drone kept draining while it is really back near full. This biases m̂ low.
- **Constant discharge rate.** The real battery comes from the physics model of
  BatteryPowerPlugin, with random air speed (σ = 4 m/s). Since 2026-11-24 the wind drifts over ~20 min
  (see Analytical/README.md): the mean drain still matches it (0.000675/s cruising), but over 100 s
  of flight it is now 0.88-1.41x the mean (1st-99th percentile), against 0.94-1.13x before.
- **Unknown deaths.** A drone that died without the observer knowing still enters m̂.

## What is recorded (Step 1)

The recorder only reads protocol state, so a recorded run is bit-identical to an unrecorded
one. This was checked on a seeded run: same cost (37254.89184305418) and the same result for
every drone.

### At every snapshot, for each observer drone i

From [recorder.py](swarm/recorder.py), returned by `drone_arrays()` and saved in the pickle
under `"states"`:

| Field | Shape | Meaning |
|---|---|---|
| `belief_battery` | (T, N) | `estimate_drone_battery` of every other drone j, at the snapshot time |
| `belief_charging` | (T, N) | 1 if i believes j is charging |
| `belief_age` | (T, N) | time since i's observation of j, in s |
| `belief_status` | (T, N) | status stored in i's row for j |
| `mean_battery_estimate` / `_true` | (T,) | m̂, and m from the real batteries of the other drones not really DEAD |
| `drones_charging_estimate` / `_true` | (T,) | N̂, and N from the real statuses |
| `charge_margin_estimate` / `_true` | (T,) | `charge_margin` with (m̂, N̂), and with (m, N) |

Own columns are NaN. In egocentric, which has no estimator, every belief is NaN.

### At every change of status: the return events

Saved in the pickle under `"returns"`, one list per drone. [main_test.py](swarm/main_test.py)
compares the drone's status before and after each timer and each packet:

| Transition | Event |
|---|---|
| anything → GOING_TO_BASE | `decision` |
| GOING_TO_BASE → CHARGING | `arrival` |
| GOING_TO_BASE → WAITING | `arrival` (on a full base, since 2026-11-24) |
| GOING_TO_BASE → MAPPING | `abort` (sent back to mapping on the way) |
| anything → DEAD | `death` |

WAITING → CHARGING (a pad freed) and CHARGING → MAPPING (the end of a recharge) are not
recorded.

Each event holds the time, the drone's real battery, the distance to the base, the energy to
reach it, all the swarm terms above, the previous status, and what triggered it:

| Trigger | Meaning |
|---|---|
| `own` | the drone's own mobility timer |
| `encounter` | a gossip message from another drone |
| `sent` | the other drone of an encounter chose for both (SHARE_GOTO_POSITION) |
| `charging_notification` | a drone at the base announced it is charging |
| `after_recharge` | the end of a recharge |
| `battery` | the battery check |

`sender` is the id of the other drone for message triggers.

## The metrics ([estimation_metrics.py](swarm/estimation_metrics.py))

The estimate only matters while the observer can decide, so every error is measured over the
samples where:
- the observer is MAPPING or GOING_TO_BASE;
- the target is not really DEAD;
- the observer does not believe the target is DEAD.

What it reports:
- **Battery of the others**, estimate − truth: bias, MAE, RMSE, 90th percentile of |error|,
  also split by the target's real status.
- **Age of the information**: median and max.
- **Charging prediction**, as a classification: recall (of the samples where the target really
  charges, how many the observer gets) and precision.
- **m̂ − m**: bias, MAE, RMSE.
- **N̂ − N**: share of samples exact, over and under.
- **Margin with the estimate − margin with the truth**: bias, MAE, and the share of samples
  where they differ by more than 0.05. That is about 68 s of flight at `DISCHARGE_RATE`.
- **Trips to the base**: each `decision` closed by the next event (arrival, abort, death, or
  open when the run ends). Battery at decision and at arrival, outcomes, triggers, deaths.

The same functions feed the report of `main_test.py` and both plot scripts, so the numbers
match everywhere.

## How to run and plot

From `src/`, on the host:

```bash
export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen
python3 -m Analytical.swarm.main_test --duration 10000 --drones 5 --transmission-range 200 \
    --seed 1 --quiet --individual <9 genes> --logs_directory <DIR>
```

The tuned individual of each range is the last `Parameters:` line of
`swarm/ga_logs/ga_logbook_<range>.txt`. The report ends with the section
"Battery estimation and returns to the base".

From `swarm/plots/`:

```bash
python3 plot_estimation.py --directory <DIR> [--drone k] [--output PREFIX]
python3 plot_returns.py    --directory <DIR> [--output NAME]
```

Everything works the same from `egocentric/`, where only the returns part applies.

### The figures

All of them use the Okabe-Ito palette, and the marker or line style also carries every
distinction.

**`plot_estimation.py`, overview** (`PREFIX_overview.png`):

| Panel | Shows |
|---|---|
| A | believed vs real battery of the others; the marker gives the target's real status |
| B | RMSE and bias against the age of the information, with a 10–90% band |
| C | distribution of N̂ − N |
| D | margin with the estimate vs margin with the truth, with the ±0.05 band |
| E | m̂ − m per drone over time |
| F | N̂ − N per drone over time |

**`plot_estimation.py`, one drone** (`--drone k`, `PREFIX_drone_k.png`): one row per other
drone j, with:
- j's real battery and what k believes it is;
- bands for j's real status;
- a strip where k believes j is charging.

**`plot_returns.py`** (`NAME.png`):
- the battery of each drone at every decision to go to the base, and its battery on arrival;
- the battery at decision against the distance to the base;
- one row per drone with its battery over time, status bands and the decisions marked on it.

The marker shape gives the outcome (arrived, aborted, died, still open) or the trigger.

## Pilot results

One seed (1), base at (0, 0), 10,000 s, swarm, the tuned individual of each range. These
are single runs: they show the trend, not statistics. All six runs had no dead drones, every
drone recharged, and coverage was 100%.

**How good the estimate is** (samples where the observer decides):

| Range | Drones | Battery RMSE | Median info age | Charging recall | N̂ exact | Margin off by > 0.05 |
|---|---|---|---|---|---|---|
| 200 m | 5 | 0.206 (bias −0.046) | 297 s | 35% | 64% (under 27%) | 31% |
| 200 m | 10 | 0.183 (bias −0.055) | 250 s | 35% | 48% (under 46%) | 38% |
| 500 m | 5 | 0.096 | 94 s | 69% | 81% | 20% |
| 500 m | 10 | 0.065 | 71 s | 71% | 73% | 9% |
| 2000 m | 5 | 0.017 | 1 s | 99% | 100% | 0% |
| 2000 m | 10 | 0.017 | 1 s | 99% | 99% | 1% |

**When the drones go to the base:**

| Range | Drones | Decisions | Aborted | Battery at decision (mean ± std) |
|---|---|---|---|---|
| 200 m | 5 | 96 | 41 | 0.50 ± 0.14 |
| 200 m | 10 | 181 | 53 | 0.55 ± 0.12 |
| 500 m | 5 | 140 | 83 | 0.53 ± 0.09 |
| 500 m | 10 | 243 | 111 | 0.58 ± 0.07 |
| 2000 m | 5 | 102 | 49 | 0.45 ± 0.08 |
| 2000 m | 10 | 363 | 254 | 0.48 ± 0.04 |

Egocentric smoke test (500 m, 4 drones, 3000 s): decisions at 0.23 ± 0.03, arrival at about
0.17, no aborts.

### What the pilot shows

- **The error depends on the age of the information.** It stays small up to about 300 s,
  then grows. Range matters through age: at 2000 m everybody hears everybody.
- **The bias is the model mismatch.** It is negative at 200 m because recharges are missed:
  the projection keeps MAPPING drones flying down to 0.20, but they really go home at about
  0.5. The one-drone figure shows the belief draining while the real battery jumps back to full.
- **At 200 m the swarm often undercounts the drones charging** (27–46% of the samples), so the
  congestion term is often blind. It sees only 35% of the real charging.
- **Swarm drones choose the base at about 0.45–0.58 battery**, well above the 0.20 the
  estimator assumes, and arrive with as little as 0.15.
- **30–70% of return decisions are aborted, and not because of the estimate.** At 2000 m with
  10 drones, 221 of the 254 aborts were `sent` by the other drone of an encounter, a median of
  4 s after the decision, with perfect information. At 200 m with 5 drones most aborts
  (37 of 41) came from a `charging_notification`.

## Open issues (to settle before Step 2)

1. **The battery drains while a drone is charging.** The battery plugin is not shut down at
   the base, so the drone keeps losing hover power: about 0.0009 per s, about 0.08 over a
   100 s charge.
   - Batteries reached 0.068 on the base, below the DEAD threshold of 0.10. Nothing notices,
     because the battery check doesn't run while CHARGING.
   - The plugin raises at 0 J, so a longer charge could crash a run.
   - It is the whole +0.045 bias on charging targets at 2000 m: the estimator assumes a
     constant battery while charging.
   - A fix (e.g. shut the plugin down at arrival and resume it at `recharge_done`) changes
     behaviour and probably the tuning.
   - **Fixed 2026-11-24**, with the pads and the waiting room of the base
     (the base node, `<system>/recharge_base.py` since 2026-11-25): the plugin is shut down at landing and resumed at
     `recharge_done`, in every system, so the battery is constant on the ground, as the
     estimator assumes. The numbers above were measured before.
2. **Aborts by `sent`.** The likely cause is the pair decision (`choose_two_cells`, which uses
   the deciding drone's view of the swarm) disagreeing with the drone's own decision. Not
   investigated.
3. **One seed only.** The numbers need many seeds before they go in the paper.

## Next steps

### Collect the scalars over many seeds

The `main_test` pickles are too large (~160 MB each) for 30+ seeds. Add the scalars of
`summarize()` to `multiple_runs.py` instead.

### Step 2: measure the influence with a controlled ablation

Keep the controller and the genes fixed, and change only the battery information the
controller gets. Add a switch in `aproximate_battery_consumption_model`:

| Condition | m and N given to the controller | What the gap shows |
|---|---|---|
| No info | m = own battery, N = 0 (swarm terms off) | the floor |
| Stale | last gossiped value, not projected | what the projection model adds |
| **Estimate** | the current system (the default) | the system under test |
| Oracle info | the true values read from the other drones | the upper bound for this controller |

- **Oracle info:** pass a shared dict through the factory, the way `results_aggregator` is
  passed. Each drone registers its protocol there and the oracle reads the real battery and
  status from it.
- **Default** stays "estimate", so current runs stay bit-identical.
- **Paired seeds.** The seed fixes the wind, the initial destinations and the base, so all
  conditions on the same seeds are paired and the Wilcoxon test of A1.md applies.

Headline number, one per range:

```
share recovered = (J_noinfo - J_estimate) / (J_noinfo - J_oracle)
```

"The gossip estimator recovers X% of the benefit of perfect battery information at 200 m,
and Y% at 2000 m."

Reduced grid: 3 ranges × {5, 10} drones × 30 seeds × 4 conditions = 720 runs, about one
night.

**Caveats:**
- The genes were tuned with the estimator, so oracle info with the same genes measures what
  the error costs *this tuned controller*. The GA may have learned to compensate for the bias,
  so oracle info could even do worse, which would be a finding in itself. If the budget allows,
  retune the oracle condition at one or two ranges.
- This oracle is not the one in the A1.md ladder (a central planner that assigns time slots).
  Oracle info with decentralized decisions sits between that planner and the swarm. It
  separates the cost of imperfect information from the cost of deciding without a central planner.

### Step 3 (optional): link the error to the outcome

- **Decision flip rate:** at every decision, also evaluate the base fitness with the true m
  and N. Split the flips into late returns (risk of dead drones) and early returns (lost
  coverage, congestion).
- **Dose–response:** oracle info plus Gaussian noise σ ∈ {0, 0.02, 0.05, 0.1, 0.2} on m, and
  random miscounts on N. Plot the cost against σ to see how much error the controller
  tolerates.
- **Model mismatch:** `DISCHARGE_RATE` ×0.8 and ×1.2 in the estimator only.

### Not done yet

The copies in `expanding_testing/` (multiple bases) don't have any of this.
