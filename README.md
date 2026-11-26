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

## Running a batch on the server

The server is reached with `ssh -p 22222 wenisbelle@139.82.47.158` and the repository
is in `~/Master_Thesis` there. Start the batch with `nohup` so it survives the ssh session
closing, and keep its output in a log:

```bash
cd ~/Master_Thesis/src
nohup python3 Analytical/multiple_runs.py --system naive egocentric swarm centralized \
    --runs 50 --drones 3 6 9 12 --range 200 --hand-tuned \
    >> multiple_runs.log 2>&1 &
```

If the server goes down, run the same command again **without `--overwrite`**. The runs
already written are skipped and it carries on from there. The runs that were still
simulating start again from t = 0.

`--overwrite` clears each directory only when the loop gets to it, so after a crash some
directories hold the new batch and others the old one. To start a batch fresh, delete its
directories by hand instead and run without `--overwrite`:

```bash
rm -rf Analytical/{naive,egocentric,swarm,centralized}/multiple_runs_hand_tuned/200/{3,6,9,12}
```

## Copying the results from the server

The server has no `rsync`, so the results come over `ssh` as a `tar` stream: one
connection, one password prompt. Run these on your own computer, from the repository root.
The download goes to `results.tgz` first, and nothing local changes unless it succeeded.

**Replace local directories with the server's copy.** The local directories are deleted and
extracted again from the server, so runs of an older batch cannot mix with the new ones in
the plots. Change the systems and the `200` to the batch you want:

```bash
for s in naive egocentric swarm centralized; do
  ssh -p 22222 wenisbelle@139.82.47.158 \
      tar -czvf - -C Master_Thesis/src/Analytical $s/multiple_runs_hand_tuned/200 > $s.tgz \
  && gzip -t $s.tgz \
  && rm -rf src/Analytical/$s/multiple_runs_hand_tuned/200 \
  && tar -xzf $s.tgz -C src/Analytical && rm $s.tgz \
  || { echo "== $s failed"; break; }
done
```

Each system comes in its own connection (about 440 MB of runs, ~100 MB compressed, for
4 drone counts × 50 runs), and the file names scroll past as they are sent. A system's
local directory is only deleted once its download is complete (`gzip -t`). If the
connection drops, run the loop again with only the systems that are left. Without a key
(`ssh-copy-id`, below) it asks for the password once per system.

**Copy every result directory** (`multiple_runs*/` of every system, including
`expanding_testing`). Files from the server overwrite the local ones with the same name.
Files that only exist locally are kept, and nothing else in the repository is touched:

```bash
ssh -p 22222 wenisbelle@139.82.47.158 \
    'cd Master_Thesis/src/Analytical && find . -type d -name "multiple_runs*" -prune | tar -czf - -T -' \
    > results.tgz && tar -xzf results.tgz -C src/Analytical && rm results.tgz
```

To see what would come before extracting, stop after the download and list it with
`tar -tzf results.tgz`. If `tar` reports `Permission denied`, the local files were written
by the docker container as root: `sudo chown -R $USER: src/Analytical` and extract again.
Run `ssh-copy-id -p 22222 wenisbelle@139.82.47.158` once to log in with a key instead of
the password.
