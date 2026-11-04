# Expanding testing: several recharge bases

Test bench for claims that go beyond the scenario the protocols were tuned on.
It holds the two systems of `Analytical/`, each in its own directory:

- `swarm/`
- `egocentric/`

Each one is a copy of the test setup of the system with the same name in
`Analytical/`, extended to **any number of recharge bases**. The map size, the
communication range, the bases, the charging capacity and the individual are
set on the command line. There is no GA here: the individuals come from the
tuning in `Analytical/<system>/main.py` (the `Parameters:` line of a logbook
in `Analytical/<system>/ga_logs/`).

With a single base, the decisions of both systems are the same as in
`Analytical/`. Given the same inputs, the fitness and the battery estimate
return exactly the same values. Whole runs still diverge after a few seconds:
gradysim orders the events of the same instant by heap position only, and
the base node adds events of its own (see [Checks](#checks)).

## What changes with several bases

### Every energy term is taken from the nearest base

| Term (`fitness.py`) | Base used |
|---|---|
| Energy effect of a cell (`energy_effect`) | the base nearest to **that cell**: the energy to fly to the cell and from there back to the closest base |
| Attraction of the base (`recharge_base_fitness`) | the base nearest to **the drone** |
| Destination when going to recharge | the cell of the base nearest to the drone (`nearest_base_cell`) |
| Congestion term of the charge margin (swarm only) | **all the bases together**, see below |

In an encounter each of the two drones gets its own nearest base.

### Congestion is counted over all the bases

The point of the capacity is to keep a minimum number of drones mapping, so
congestion is the number of drones charging **at once over all the bases**,
whatever base they are on. `Nc` (`--charging-capacity`, default `int(drones / 3)`,
the capacity of the single base of `Analytical/`) is the most drones charging
at once before the protocol and the cost react. Future base limitations will be explored
in the next works. The rule is the same as in
`Analytical/`; only the drones are now spread over several bases:

- the swarm drone's estimate of how many others are charging
  (`aproximate_battery_consumption_model`) is exactly the original one;
- the congestion cost is `congestion_penalty_rate` of the total excess over `Nc`.

Both are soft: they push the drones to stay out longer when the swarm is
congested, they do not forbid more than `Nc` drones charging at once. The
report says how much of the time `Nc` was respected.

The occupancy of **each** base is still recorded and reported, to see where
the drones charge, but it enters neither the decisions nor the cost.

### The recharge base is a node

Each base is a node of the simulation (`recharge_base.py`), like the drones,
but it never moves. The bases are added after the drones, so drones keep the
ids `0..n-1` and base `b` is node `n + b`.

- **Plugging in.** A drone that reaches a base plugs into it
  (`ChargingRegistry.dock`), and unplugs when its recharge ends. The base always
  knows who is charging on it.
- **Shared knowledge between bases.** All the bases of a simulation share one
  `ChargingRegistry`. In reality that would be a second radio, long range and
  low bandwidth; here it is a shared variable. Each base therefore knows exactly
  how many drones charge on every base.
- **Heartbeat (swarm only).** Every 1 s each base broadcasts a
  `BaseHeartbeatMessage` with the drones charging on **every** base: drone id,
  time it reached the base, battery then. It has the same range as every other
  message. It replaces the `ChargingNotificationMessage` the charging drone sent
  itself in `Analytical/swarm`. A drone in range stores those drones as CHARGING
  in its swarm table. When one of them is new to it, it chooses its destination
  again, the same as on the old notification. Because every base reports all of
  them, a drone near one base also learns about the drones charging on the
  others.
- **Egocentric.** The bases are the same nodes at the same places, and drones
  plug in the same way, but the bases send nothing. The egocentric drone only
  looks at its own battery.
- **Antenna altitude.** gradysim checks the range in 3D. The drones fly and
  charge at 50 m, so the base nodes sit at 50 m too (`BASE_ANTENNA_ALTITUDE` in
  `main.py`). That way a base heartbeat reaches exactly as far as a
  drone-to-drone message, and as far as the old notification sent by a drone
  sitting on the base. On the ground it would reach only
  `sqrt(range² - 50²)` horizontally, 87 m for a 100 m range.

### Scenario

- The drones take off from the bases in turn: drone `i` from base
  `i % number_of_bases` (`protocol.take_off_base`).
- Each base is moved to the centre of its cell, the point the drones land on.
  `main_test.py` prints it when it moves one. Two bases on the same cell are
  rejected.
- The map is square (`--map-size N` cells, 20 m apart). The protocol shares one
  centre offset between both axes, so a rectangular map would put the cells in
  the wrong place.

## Layout

```
expanding_testing/
├── README.md
├── plot_charging_bases.py global uncertainty and drones charging per base, several runs
├── multiple_runs_bases.py                    many runs of the multi-base scenarios, one range
├── run_all_tests_with_multiple_runs_bases.py the same for every tuned range
├── plot_multiple_runs_bases.py               plots of those runs, per scenario and range
├── swarm/
│   ├── main.py            scenario, simulation and cost (no GA)
│   ├── main_test.py       single simulation, with checks, log and recording
│   ├── protocol.py        drone
│   ├── recharge_base.py   base node and ChargingRegistry
│   ├── fitness.py
│   ├── recorder.py        writes the recording the plots read
│   ├── visualization.py
│   ├── logs/              default output of main_test.py
│   └── multiple_runs/     output of multiple_runs_bases.py, <range>/<scenario>
└── egocentric/            same files
```

`diff Analytical/swarm/X expanding_testing/swarm/X` shows exactly what the
multi-base extension changed. `diff expanding_testing/swarm/X
expanding_testing/egocentric/X` shows the same energy-only differences as in
`Analytical/`.

## Running a test simulation

From the `src` directory, with gradysim on the `PYTHONPATH`:

```bash
cd src
export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen

# one base at the centre, as in Analytical/swarm
python3 -m Analytical.expanding_testing.swarm.main_test

# two bases, 5 drones, 4000 s
python3 -m Analytical.expanding_testing.swarm.main_test --drones 5 --duration 4000 \
    --base -500 -500 --base 500 500 --logs_directory logs_2bases

# three bases, at most 2 drones charging at once, bigger map, 500 m range (and its tuned individual)
python3 -m Analytical.expanding_testing.swarm.main_test --map-size 150 --transmission-range 500 \
    --base -800 -800 --base 800 -800 --base 0 800 --charging-capacity 2

# any other individual
python3 -m Analytical.expanding_testing.swarm.main_test --individual 90.6 95.4 ...

# the same for egocentric (6 genes instead of 9)
python3 -m Analytical.expanding_testing.egocentric.main_test --base -500 -500 --base 500 500
```

| Flag | Default | Description |
|---|---|---|
| `--base X Y` | one base at `(0, 0)` | position of a recharge base, in m, map centre at `(0, 0)`. Repeat it for every base |
| `--charging-capacity NC` | `int(drones / 3)` | most drones charging at once, over all the bases together, before the congestion terms start |
| `--map-size N` | `100` | side of the square map in cells, 20 m apart (100 → ±1000 m) |
| `--transmission-range M` | `200` | communication range of the drones and of the bases, in m. Also picks the tuned individual |
| `--individual G1 G2 ...` | the one tuned for `--transmission-range` | genes in the `GENE_BOUNDS` order: 9 for swarm, 6 for egocentric |
| `--drones N` | `10` | number of drones |
| `--duration SECONDS` | `10000` | simulated seconds |
| `--logs_directory NAME` | `logs` | directory, relative to `<system>/` (or absolute), for the log and the recording |
| `--monitor-interval SECONDS` | `10.0` | simulated seconds between two samples of the drone state in the recording |
| `--sample-interval SECONDS` | `1.0` | simulated seconds between two samples of the global map, the integration step of the cost |
| `--quiet` | off | train mode, without the per-routine log (and without the distance flown) |
| `--map-plot` | off | live per-drone map (very slow) |
| `--simulation-plot` | off | gradysim 3D view (slow), shows the bases as nodes |

The report is that of `Analytical/<system>/main_test.py`, plus:

- a **per base** column in the drone table: recharges at each base, e.g. `3/1`;
- a **Recharge bases** section: recharges, most drones charging at once and
  mean occupancy, for each base and for all of them, plus the time with more
  than `Nc` drones charging and the share of the run `Nc` was respected;
- the split of the cost into its terms (uncertainty, dead, charging).

The exit status is 0 when every check passes and 1 otherwise.

## Plotting a run

The recording has the format of `Analytical/<system>`, plus the base of each
drone (`charging_base` series) and the base positions (`meta["base_positions"]`).
The plot scripts of `Analytical/<system>/plots` read it. Give them the full
path, which `main_test.py` prints at the end of the run:

```bash
cd src/Analytical/swarm/plots
python3 plot_drones.py --directory /full/path/to/expanding_testing/swarm/logs_2bases
```

`plot_more_metrics.py` draws the drones charging on all the bases together
against `Nc`, which is the congestion as it is counted here.

### Charging per base, comparing runs

`plot_charging_bases.py`, in this directory, compares runs on the global
uncertainty and on the drones charging **on each base**, stacked so the top
is the total that `Nc` limits, with the time above `Nc` shaded. Each run is a
logs directory and its caption. A relative directory is looked up here first,
then from the current directory, so the single-base runs of `Analytical/` are
`../swarm/logs` or `../egocentric/logs` (all their charging drones count on
base 0):

```bash
cd src/Analytical/expanding_testing
python3 plot_charging_bases.py --run swarm/logs_2bases "Swarm, 2 bases" \
    --run egocentric/logs_2bases "Egocentric, 2 bases" \
    --run ../swarm/logs "Swarm, conventional" --per-cell
```

| Flag | Description |
|---|---|
| `--run DIRECTORY CAPTION` | a run to plot, repeat it for every run (up to 6) |
| `--per-cell` | global uncertainty divided by the number of cells, needed when the maps have different sizes |
| `--output FILE` | save the figure (png, pdf, svg ...) instead of opening a window |

It also prints the same numbers as a table, and where the bases of each run
are. The time above `Nc` comes from the recording samples (`--monitor-interval`,
10 s by default), so it can differ a little from the report of `main_test.py`,
which uses the 1 s samples.

## Multiple runs on the multi-base scenarios

`multiple_runs_bases.py` is `Analytical/multiple_runs.py` for the multi-base
tests: it runs the systems many times (20 by default) with the individual the
GA tuned for the range, each run with its bases placed again, and keeps the
swarm level series of every run. `run_all_tests_with_multiple_runs_bases.py`
runs it for every tuned range, one after the other, and
`plot_multiple_runs_bases.py` draws the figures of
`Analytical/plot_multiple_runs.py`, one pair per scenario and range, with
swarm and egocentric on the same figures.

| Scenario | Map | Drones | Bases | `Nc` |
|---|---|---|---|---|
| `100x100_10drones_2bases` | 100x100 cells (2000 m) | 10 | 2 | 3 |
| `100x100_10drones_3bases` | 100x100 cells (2000 m) | 10 | 3 | 3 |
| `200x200_20drones_2bases` | 200x200 cells (4000 m) | 20 | 2 | 6 |
| `200x200_20drones_3bases` | 200x200 cells (4000 m) | 20 | 3 | 6 |

**Base placement.** The map is split into its 4 quadrants. Each base takes a
quadrant of its own, drawn at random among the free ones, and its position is
drawn from a Gaussian centred on the centre of that quadrant, sigma a third of
half the side of the quadrant (3 sigma reaches its border: 167 m on the 100
cell map, 333 m on the 200 cell one), snapped to the nearest cell centre and
drawn again when it falls outside the quadrant.

Run k is seeded with `--seed + k`, and the bases are drawn from that seed too.
Run k therefore has the same bases and wind on swarm and egocentric, and on
every range, and the 3-base scenario of run k is its 2-base scenario plus one
base.

The individual is the `Parameters:` line of
`Analytical/<system>/ga_logs/ga_logbook_<range>.txt`. A range without one, for
any of the systems given, is an error and nothing is run.

```bash
cd src
export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen

# every scenario, both systems, 200 m, 20 runs each
python3 Analytical/expanding_testing/multiple_runs_bases.py --system swarm egocentric

# every tuned range (100 ... 2000 m), one after the other
python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm egocentric

# only some ranges and scenarios
python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm \
    --ranges 100 200 --scenario 100x100_10drones_2bases 100x100_10drones_3bases

# figures: 200 m, every scenario, on screen
python3 Analytical/expanding_testing/plot_multiple_runs_bases.py
# every range, saved as figures/bases_<range>_<scenario>_{uncertainty,charging}.pdf
python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --range 100 200 500 750 1000 2000 \
    --save figures/bases.pdf --no-show
```

The runs are written to `<system>/multiple_runs/<range>/<scenario>`
(`--logs-root` changes `multiple_runs`) with the files of
`Analytical/multiple_runs.py`: `run_<k>.pkl`, `config.json` and `summary.txt`.
Each run also keeps the drones charging on each base (`charging_per_base`) and
the quadrant of each base. The sweep writes `ranges_summary.txt` to
`<system>/multiple_runs`. Every run of every system and scenario of a range
goes into one pool of processes, the longest first. Runs already on disk are
skipped, `--overwrite` runs them again, and a directory holding runs of
another scenario (another duration, seed, individual ...) is refused.

## Checks

These are the checks run when this was written (2026-11-02):

- **One base.** On 300 random decisions (one drone and two drones) the
  fitness returns exactly the same as `Analytical/swarm` and
  `Analytical/egocentric`. The swarm battery estimate is the original function,
  and it matched on 2000 random swarm tables. Whole runs from the same seed
  still diverge within seconds, because of the event ordering of gradysim
  mentioned above.
- **Several bases.** `energy_effect` matches a brute-force minimum over three
  bases.
- **Swarm, 2 bases, 5 drones, 4000 s, the hardcoded `BEST_INDIVIDUAL` of `Analytical/swarm/main.py`.** Every check passed;
  17 recharges, 12 on one base and 5 on the other, and `Nc` = 1 was respected
  97.5% of the time.
- **Swarm, base heartbeats.** In an instrumented run, the drones learned 14
  times about a drone charging on a base other than the one nearest to them
  (shared knowledge between bases), and chose their destination again each time.
