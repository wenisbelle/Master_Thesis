# Naive: the egocentric drone without an energy model

`naive/` is the bottom rung of the baseline ladder in [`A1.md`](../../../A1.md):
**naive** → egocentric → swarm → centralized. It answers the question "does being
energy-aware matter at all?". It is the egocentric drone with its energy logic replaced
by the simplest rule there is:

- **Where to map:** chosen by the uncertainty alone. The energy factor of the cells
  (Eq. 14) is always 1, whatever the battery and the distance to the base.
- **When to recharge:** a MAPPING drone goes to the base as soon as its battery drops
  below `RETURN_BATTERY_THRESHOLD` = 30%. It checks on every mobility tick (1 s).

Everything else is egocentric's: the maps merged in encounters, the heartbeat and
gossip, the pair decisions (only between two MAPPING drones), the camera, the
mobility and braking approach, the battery, wind and seeds, the 300 s recharge, death
at 10%, the cost, the GA, the monitors, the recorder and the plots.

The code as of 2026-11-15 is the source of truth. Where this file and the code disagree,
the code is right.

## What differs from egocentric

| | egocentric | naive |
|---|---|---|
| Energy factor of the cells (Eq. 14) | `exp(γ · min(R, 0))`: cells it cannot fly to and back from are damped | **1**: there is no `energy_effect`, the weights of `fitness.cells_priority` are φ alone |
| Battery of the others | a table per drone (battery, time, status), merged in the gossip, plus a DEAD notification broadcast by dead drones | none: nothing would read it. The gossip carries the map, position and status only, and a dead drone stays silent |
| Going to the base | when the base attraction `κ·exp(−γ·R_safe)` beats the best cell | when **battery < 0.30**, checked every second (`protocol.go_to_base`) |
| Reserve for the trip home | tuned `charge_margin` plus the energy to reach the base | none. It only works because 30% is enough to cross the map: about 190 s of flight corner to corner, 0.14 of battery |
| Pair decision | `choose_two_cells` can send a drone to the base | the same `choose_two_cells`, but both drones always map. The drone below 30% goes on its next tick |
| Genes | 6 | **3**: `base_variance`, `alpha_variance_modifier`, `distance_between_drone_norm` |
| `INITIAL_UNCERTAINTY` | 0.5 | **1.0**, as swarm and centralized (A1.md asks for the same value in every system; egocentric is the one left at 0.5) |

`fitness.py` has no `recharge_base_fitness` and no battery inputs. Its `choose_one_cell`
and `choose_two_cells` only take the map, the positions and the speed, and only return
cells to map. `Nc` and the measured `DISCHARGE_RATE` are not decision parameters either:
`main.py` keeps them for the cost and the plots (`charging_base_capacity`, the reference
line of `plot_returns.py`). Since 2026-11-24 `Nc` is also the number of pads of the base
(the base node, `naive/recharge_base.py`), as in every system: the naive drone does not look at it before
going, but a drone that lands with every pad taken waits on the ground, battery frozen,
until one frees. `main_test.py` labels a trip to the base as `threshold` in the
returns, where swarm's says `own`.

## The individual

Naive has not been tuned yet. For each range, `main.default_individual(range)` returns:
- the `Parameters:` line of `naive/ga_logs/ga_logbook_<range>.txt`, once it exists;
- otherwise **the coverage genes of the egocentric individual tuned for the range**
  (`egocentric/ga_logs/ga_logbook_<range>.txt`, picked by gene name).

So naive and egocentric map with the same genes, and differ only in the energy decision.
`multiple_runs.py`, `all_tests_distance.py` and `run_all_tests_with_multiple_runs.py` use
it. `BEST_INDIVIDUAL` in `main.py` is the 200 m one.

To tune it, use the GA of egocentric with 3 genes, one run per range:

```
python3 Analytical/all_optimizations_distance.py --system naive
# then move naive/ga_logbook_<range>.txt into naive/ga_logs/
```

## Running

From `src/`, with gradysim on the path (see [`../README.md`](../README.md#running)):

```
python3 -m Analytical.naive.main_test --drones 5 --seed 1                  # one recorded run, report + checks
python3 -m Analytical.naive.main_test --hand-tuned                         # HAND_TUNED_INDIVIDUAL of main_test.py
python3 Analytical/multiple_runs.py --system naive --runs 20 --range 200 --drones 10
python3 Analytical/plot_multiple_runs.py --system naive egocentric swarm centralized --range 200 --drones 10
python3 Analytical/all_tests_distance.py --system naive --ranges 200 --drones 5
```

- The plots of one run work from `naive/plots/` as in the other systems.
- In `plot_multiple_runs.py` naive is drawn in reddish purple, with a dash-dot line and
  hexagon markers.

## Checked on 2026-11-15

- **`main_test.py`**, 5 drones, 3,000 s, seed 1: every check passed.
  - All 10 trips to the base were decided at a battery of 0.300 and arrived with
    0.22–0.27.
  - No drone died.
- **`multiple_runs.py`**, 2 runs of 1,500 s with 3 drones: the run files, summary and
  figures are the same as for the other systems.
  - All 3 drones charged at the same moment in both runs. They all start full and reach
    30% together, and nothing staggers them. That is the congestion the energy-aware
    systems are meant to avoid.
- **The 7 plot scripts** in `plots/` are egocentric's, unchanged. All of them ran on the
  recording of that `main_test.py` run.
