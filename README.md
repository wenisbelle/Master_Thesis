# Analytical coordination protocol

Two variants of the protocol, with the same layout and the same test and plot
workflow:

- `swarm/`
- `egocentric/`

Everything below applies to both. Replace `swarm` with `egocentric` in the
commands to work on the other one.

## Layout

```
swarm/
├── main.py            GA tuning and the simulation itself
├── main_test.py       single simulation, with checks, log and recording
├── protocol.py
├── fitness.py
├── recorder.py        writes the recording the plots read
├── plots/             plot scripts, shared by every logs directory
│   ├── plot_drones.py
│   ├── plot_metrics.py
│   ├── plot_more_metrics.py
│   ├── plot_states.py
│   └── plot_decisions.py
├── logs/              default output of main_test.py
└── logs_<name>/       one directory per run you want to keep
```

A logs directory only holds the output of one run:

- `main_test.log`: per-routine protocol log
- `test_recording.pkl`: recording read by the plot scripts

## Running a test simulation

Run it from the `src` directory, with gradysim on the `PYTHONPATH`:

```bash
cd src
export PYTHONPATH=$PWD:$PWD/gradys-sim-nextgen

python3 -m Analytical.swarm.main_test                              # writes to swarm/logs
python3 -m Analytical.swarm.main_test --logs_directory logs_3      # writes to swarm/logs_3
python3 -m Analytical.swarm.main_test --duration 1000 --drones 2 --logs_directory logs_short
```

The logs directory is resolved relative to `swarm/` (where `main_test.py` is),
whatever the current directory is, and is created if it does not exist. Running
again with the same directory overwrites the previous run in it.

| Flag | Default | Description |
|---|---|---|
| `--logs_directory NAME` | `logs` | directory, relative to `swarm/`, where the log and the recording are written (`--logs-directory` also works) |
| `--duration SECONDS` | `10000` | simulated seconds |
| `--drones N` | `5` | number of drones |
| `--monitor-interval SECONDS` | `10.0` | simulated seconds between two samples of the drone state in the recording |
| `--sample-interval SECONDS` | `1.0` | simulated seconds between two samples of the global map, the integration step of the cost |
| `--quiet` | off | run in train mode, without the per-routine log |
| `--map-plot` | off | live per-drone map (very slow) |
| `--simulation-plot` | off | gradysim 3D view (slow) |

When the run ends, `main_test.py` prints where the recording was written, the
command to plot it, a summary of the run and a list of checks. The exit status
is 0 when every check passes and 1 otherwise.

## Plotting a run

Run the plot scripts from the `plots/` directory and name the logs directory of
the run with `--directory` (default `logs`):

```bash
cd src/Analytical/swarm/plots

python3 plot_drones.py                         # swarm/logs
python3 plot_drones.py --directory logs_3      # swarm/logs_3
python3 plot_decisions.py --directory logs_3 --drone 2
```

As in `main_test.py`, the directory is resolved relative to `swarm/`, so the
scripts can also be run from anywhere, e.g. `python3 swarm/plots/plot_drones.py
--directory logs_3`.

| Script | What it shows |
|---|---|
| `plot_drones.py` | per-drone battery, coverage, map uncertainty, distance flown and status timeline |
| `plot_metrics.py` | uncertainty of each drone against the global map, and the cells no drone has seen |
| `plot_more_metrics.py` | drones charging at once against the base capacity, and the effective charge margin of each drone |
| `plot_states.py` | slider over the uncertainty map of every drone and the global map |
| `plot_decisions.py` | slider over the changes of destination of one drone (`--drone ID`, default 0) |

Every script accepts `--directory`, and `--help` lists the options.
