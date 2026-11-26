# Expanding testing: several recharge bases

Test bench for claims that go beyond the scenario the protocols were tuned on.
It holds three systems of `Analytical/`, each in its own directory:

- `swarm/`
- `egocentric/`
- `naive/` (added 2026-11-18)

Each one is a copy of the test setup of the system with the same name in
`Analytical/`, extended to **any number of recharge bases**. The map size, the
communication range, the bases, the charging capacity and the individual are
set on the command line. There is no GA and no cost here. The individuals are
the ones `Analytical/<system>` runs:
- swarm and egocentric: the `Parameters:` line of a logbook in
  `Analytical/<system>/ga_logs/`;
- naive: `naive.main.default_individual`, which uses the coverage genes of the
  egocentric logbook until naive is tuned;
- with `--hand-tuned`: `HAND_TUNED_INDIVIDUAL` of `Analytical/<system>/main_test.py`.

A run is looked at through the global uncertainty and the drones charging and
dead over time.

With a single base, the decisions of all three systems are the same as in
`Analytical/`. Given the same inputs, the fitness and the battery estimate
return exactly the same values. Whole runs still diverge after a few seconds:
gradysim orders the events of the same instant by heap position only, and
the base node adds events of its own (see [Checks](#checks)).

**Synced with `Analytical/` on 2026-11-18.** The copies were made on
2026-11-02, so two later changes of `Analytical/swarm` and `egocentric` were
ported then:
- **Base utility, Eq. (15).** kappa = K · mean(I/tau) (`fitness.mean_efficiency`),
  no longer the constant K.
- **`INITIAL_UNCERTAINTY = 1.0`.** It was 1.0 in swarm and 0.5 in egocentric, then
  10.0 in all three; since 2026-11-26 it is 1.0 in every Analytical system.

GA logbooks written before 2026-11-17 hold K in the old units, so with the new
kappa their individuals are not tuned anymore. Until they are tuned again, use
`--hand-tuned`.

### Naive

`naive/` is `Analytical/naive` with several bases. The cells are chosen by the
uncertainty alone. A MAPPING drone goes to the **base nearest to it** when its
battery drops below `RETURN_BATTERY_THRESHOLD` = 30% (25% for a few hours on 2026-11-24), plugs in there like the
others, and the bases send nothing to it, as for egocentric. `Nc` is only the
reference the runs are measured against, because the naive drone never reads it.
The genes are the 3 coverage genes of `Analytical/naive`.

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
at once before the protocol reacts. Future base limitations will be explored
in the next works. The rule is the same as in `Analytical/`, only the drones
are now spread over several bases: the swarm drone's estimate of how many
others are charging (`aproximate_battery_consumption_model`) is exactly the
original one.

It is soft: it pushes the drones to stay out longer when the swarm is
congested, it does not forbid more than `Nc` drones charging at once over all
the bases. The report says how much of the time `Nc` was respected. What is
hard is the number of pads of each base, see below.

### Pads and waiting room of every base (since 2026-11-24)

Every base has its own pads and its own waiting room, and manages them alone, as the
single base of `Analytical/` does (the base node of each system's `recharge_base.py`):
- **How many.** `PADS_PER_BASE` in each `main.py`; `None`, the default, gives every
  base `Nc` pads, the capacity of the single base of `Analytical/`.
- **A full base.** A drone that lands with every pad of that base taken waits on the
  ground next to it, in that base's waiting room (status WAITING), and takes the first
  pad that frees there, in the order the drones landed. It does not move on to
  another base.
- **The battery on the ground.** Waiting or charging, the motors are off: the battery
  plugin is stopped from landing to take-off, so the battery does not change and no
  drone can die at a base.
- **Kept apart from `Nc`.** The pads are per base, `Nc` is still the soft reference
  over all the bases. With the default, several bases together have more pads than `Nc`.
- **Report.** `main_test.py` has a "Pads and waiting rooms of the bases" section and a
  check that no base ever had more drones charging than its pads.

The occupancy of **each** base is still recorded and reported, to see where
the drones charge, but it does not enter the decisions.

### The recharge base is a node

Each base is a node of the simulation, like the drones, but it never moves. It is
the node of the system's own `recharge_base.py` (`expanding_testing/<system>/`, since
2026-11-25; from 2026-11-24 one shared `Analytical/recharge_base.py`, see
`Analytical/base_node.md`). The bases are added after the drones, so drones keep the
ids `0..n-1` and base `b` is node `n + b` (`base_node_id`).

- **The pads, by messages.** A drone that lands on a base is WAITING and sends that
  base `LANDED`. The base answers `PAD` at once when a pad is free and nobody waits,
  otherwise when the drone's turn in the waiting room comes. At take-off the drone
  sends `TAKE_OFF`, and the base gives the pad to the first drone of its waiting room
  at that instant. Each base owns its pads and its waiting room, the drones never
  touch them. The reports read them from the nodes (`global_map.recharge_bases`).
- **No shared knowledge between bases (since 2026-11-25).** A base knows only who is
  on it. Until 2026-11-25 the bases shared the list of every base's pads (a "second
  radio" between them, a shared variable in `main.py`) and each one knew who was on
  every base.
- **Heartbeat (swarm only, `BROADCAST_BASE` in `main.py`).** Every 1 s, while a drone
  is on its ground, a base broadcasts a `BaseHeartbeatMessage` with its pads, the
  drones charging on it (id, time the charge started, battery) and the drones of its
  waiting room (id, time their charge will start, battery, worked out by the base). It has
  the same range as every other message. A drone in range stores the charging ones as
  CHARGING and the waiting ones as WAITING, with the time their charge will start, in
  its swarm table, and counts both as on a base. When one of them is new to it, it
  chooses its destination again. A drone learns about the drones on the other bases
  from the drones it meets, which pass on the rows they hold (newest wins). Until
  2026-11-25 every base reported all of them, so a drone near one base learnt about
  the others at once.
- **Egocentric and naive.** The bases are the same nodes at the same places and give
  the pads the same way, but they send no heartbeat. These drones only look at their
  own battery.
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
├── multiple_runs_bases.py                    many runs of one multi-base scenario, one range
├── run_all_tests_with_multiple_runs_bases.py the same for every tuned range
├── plot_multiple_runs_bases.py               plots of those runs, per scenario and range
├── swarm/
│   ├── main.py            scenario and simulation (no GA, no cost)
│   ├── main_test.py       single simulation, with checks, log and recording
│   ├── protocol.py        drone
│   ├── fitness.py
│   ├── recorder.py        writes the recording the plots read
│   ├── visualization.py
│   ├── logs/              default output of main_test.py
│   ├── multiple_runs/     output of multiple_runs_bases.py, <range>/<scenario>
│   └── multiple_runs_hand_tuned/  the same with --hand-tuned
├── egocentric/            same files
└── naive/                 same files
```

`diff Analytical/swarm/X expanding_testing/swarm/X` shows exactly what the
multi-base extension changed. `diff expanding_testing/swarm/X
expanding_testing/egocentric/X` shows the same energy-only differences as in
`Analytical/`. `naive/` is `Analytical/naive` with the same multi-base changes as
egocentric (`diff Analytical/naive/X expanding_testing/naive/X`), except `main.py`
and `main_test.py`, which are those of `expanding_testing/egocentric` without the
energy genes.

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

# HAND_TUNED_INDIVIDUAL of Analytical/swarm/main_test.py
python3 -m Analytical.expanding_testing.swarm.main_test --hand-tuned --base -500 -500 --base 500 500

# the same for egocentric (6 genes instead of 9) and naive (3 genes)
python3 -m Analytical.expanding_testing.egocentric.main_test --base -500 -500 --base 500 500
python3 -m Analytical.expanding_testing.naive.main_test --hand-tuned --base -500 -500 --base 500 500
```

| Flag | Default | Description |
|---|---|---|
| `--base X Y` | one base at `(0, 0)` | position of a recharge base, in m, map centre at `(0, 0)`. Repeat it for every base |
| `--charging-capacity NC` | `int(drones / 3)` | most drones charging at once, over all the bases together, before the protocol's congestion term starts (naive never reacts to it) |
| `--map-size N` | `100` | side of the square map in cells, 20 m apart (100 → ±1000 m) |
| `--transmission-range M` | `100` | communication range of the drones and of the bases, in m. Also picks the tuned individual |
| `--individual G1 G2 ...` | the one `Analytical/<system>` runs at `--transmission-range` | genes in the `GENE_NAMES` order of `main.py`: 9 for swarm, 6 for egocentric, 3 for naive |
| `--hand-tuned` | off | run `HAND_TUNED_INDIVIDUAL` of `Analytical/<system>/main_test.py` |
| `--drones N` | `10` | number of drones |
| `--duration SECONDS` | `10000` | simulated seconds |
| `--logs_directory NAME` | `logs` | directory, relative to `<system>/` (or absolute), for the log and the recording |
| `--monitor-interval SECONDS` | `10.0` | simulated seconds between two samples of the drone state in the recording |
| `--sample-interval SECONDS` | `1.0` | simulated seconds between two samples of the global map, the integration step of the accumulated uncertainty |
| `--quiet` | off | train mode, without the per-routine log (and without the distance flown) |
| `--map-plot` | off | live per-drone map (very slow) |
| `--simulation-plot` | off | gradysim 3D view (slow), shows the bases as nodes |

The report is that of `Analytical/<system>/main_test.py` without the cost, plus:

- a **per base** column in the drone table: recharges at each base, e.g. `3/1`;
- a **Recharge bases** section: recharges, most drones charging at once and
  mean occupancy, for each base and for all of them, plus the time with more
  than `Nc` drones charging and the share of the run `Nc` was respected.

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
tests. It runs the systems many times (20 by default), each run with its bases
placed again, and keeps the swarm level series of every run.
`run_all_tests_with_multiple_runs_bases.py` runs it for every tuned range, one
after the other. `plot_multiple_runs_bases.py` draws the figures of
`Analytical/plot_multiple_runs.py`, one set per scenario and range, with all the
systems on the same figures:
- the global uncertainty;
- the global uncertainty relative to a reference system (`--reference`, naive by
  default), run k against run k;
- the drones charging above `Nc`;
- the time between two visits of a cell, by the distance from the cell to its
  nearest base.

The systems are drawn under the name of the system of `Analytical/` they extend,
so swarm, egocentric and naive have the same colour, line, marker and legend label
as in `plot_multiple_runs.py`.

**The scenario** is set on the command line, one per call:

| Flag | Default | Description |
|---|---|---|
| `--cells SIDE` | `100` | side of the square map in cells, 20 m apart: `--cells 200` is 200x200 (4000 m) |
| `--drones N` | `10` | number of drones. `Nc = int(drones / 3)`, over all the bases together |
| `--bases N` (or `--base N`) | `2` | number of recharge bases, 1 to 4: one per quadrant, see below |

The runs of a scenario go to the directory `<cells>x<cells>_<drones>drones_<bases>bases`,
e.g. `100x100_10drones_2bases`, the same names as the four fixed scenarios used before.

**Base placement.** The map is split into its 4 quadrants. Each base takes a
quadrant of its own, drawn at random among the free ones, and its position is
drawn from a Gaussian centred on the centre of that quadrant, sigma a third of
half the side of the quadrant (3 sigma reaches its border: 167 m on the 100
cell map, 333 m on the 200 cell one), snapped to the nearest cell centre and
drawn again when it falls outside the quadrant.

Run k is seeded with `--seed + k`, and the bases are drawn from that seed too.
Run k therefore has the same bases and wind on every system, on every range and
with any number of drones, and the 3-base scenario of run k is its 2-base
scenario plus one base.

The individual is the one `Analytical/<system>` runs at the range (see the top of
this file), or `HAND_TUNED_INDIVIDUAL` of `Analytical/<system>/main_test.py` with
`--hand-tuned`. A system without one is an error and nothing is run.

```bash
cd src
export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen

# the three systems, hand-tuned, 100x100 map, 10 drones, 2 bases, 100 m, 20 runs each
python3 Analytical/expanding_testing/multiple_runs_bases.py --system swarm egocentric naive --hand-tuned

# 200x200 map, 20 drones, 3 bases, 500 m
python3 Analytical/expanding_testing/multiple_runs_bases.py --system swarm egocentric naive --hand-tuned \
    --cells 200 --drones 20 --bases 3 --range 500

# every tuned range (100 ... 2000 m), one after the other, same scenario on each
python3 Analytical/expanding_testing/run_all_tests_with_multiple_runs_bases.py --system swarm egocentric naive \
    --hand-tuned --cells 100 --drones 10 --bases 3

# figures: 100 m, every scenario run, on screen
python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --hand-tuned
# only some scenarios (each flag takes several values), every range, saved as
# figures/bases_<range>_<scenario>_{uncertainty,charging,revisit}.pdf
python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --hand-tuned --cells 100 --bases 2 3 \
    --range 100 200 500 750 1000 2000 --save figures/bases.pdf --no-show
```

The runs are written to `<system>/multiple_runs/<range>/<scenario>`, or
`<system>/multiple_runs_hand_tuned/<range>/<scenario>` with `--hand-tuned`
(`--logs-root` changes it), with the files of `Analytical/multiple_runs.py`:
`run_<k>.pkl`, `config.json` and `summary.txt`. Each run also keeps the drones
charging on each base (`charging_per_base`) and the quadrant of each base. The
sweep writes `ranges_summary_<scenario>.txt` to the logs root of each system.
The runs of every system go into one pool of processes. Runs already on disk are
skipped, and `--overwrite` runs them again. A directory holding runs of another
configuration (another duration, seed, individual, initial uncertainty ...) is
refused. `config.json` records `initial_uncertainty` since 2026-11-18, so runs
made before the sync with `Analytical/` are never mixed with new ones.

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

Checks run on 2026-11-18, after porting kappa and adding naive:

- **One base, after the port.** I ran 150 random decisions (one drone and two
  drones), with kappa = K · mean(I/tau). The fitness of all three systems returned
  exactly what `Analytical/swarm`, `egocentric` and `naive` return. 25 and 24 of
  those decisions chose the base, so the new kappa path was exercised.
- **Naive, 2 bases.** `--hand-tuned`, 4 drones, 1800 s, 60x60 map. Every check
  passed. Each drone flew to the base nearest to it at 30% battery and recharged
  there: 3 recharges on one base, 1 on the other.
- **`multiple_runs_bases.py`.** All three systems with `--hand-tuned`,
  60x60 map, 4 drones, 2 bases, 2 runs of 1500 s:
  - run k had the same bases on every system;
  - a second call skipped the runs already on disk;
  - another `--duration` was refused;
  - `run_all_tests_with_multiple_runs_bases.py` and `plot_multiple_runs_bases.py`
    read the runs back.
