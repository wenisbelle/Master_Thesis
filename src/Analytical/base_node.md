# The recharge base as a node of the simulation (2026-11-24)

Earlier on 2026-11-24 the base got hard pads and a waiting room
([`waiting_room.md`](waiting_room.md)). They were a shared Python object
(`ChargingStation`) that every drone called directly. Later the same day the base
became a **gradysim node**, in all seven Analytical systems. The drones now talk to it
with messages, and in swarm **the base, not the charging drones, broadcasts who is on
the ground**: the drones charging and the drones in the waiting room.

> **Since 2026-11-25 the node holds its pads and its waiting room itself, and every
> system has its own `recharge_base.py`** (`swarm/`, `egocentric/`, `naive/`,
> `centralized/` and the three `expanding_testing/` copies). With several bases, each one
> manages its landings and knows only about itself. See
> [Since 2026-11-25](#since-2026-11-25-the-pads-inside-the-node-one-copy-per-system).

## Contents

- [Before and after](#before-and-after)
- [Why](#why)
- [The messages](#the-messages)
- [Where in the code](#where-in-the-code)
- [Swarm: the base heartbeat](#swarm-the-base-heartbeat)
- [The other systems](#the-other-systems)
- [Since 2026-11-25: the pads inside the node, one copy per system](#since-2026-11-25-the-pads-inside-the-node-one-copy-per-system)
- [Checks run on 2026-11-24](#checks-run-on-2026-11-24)
- [Consequences and open points](#consequences-and-open-points)

## Before and after

| | Pads as a shared object (earlier 2026-11-24) | Base node (since) |
|---|---|---|
| The base | `ChargingStation`, handed to every drone by the protocol factory | A node, `recharge_base.RechargeBase`, added after the drones. It owns its `ChargingStation` |
| Landing | The drone called `station.land()` and got its answer at once | The drone is WAITING and sends `LANDED`. The base answers `PAD` |
| Waiting room | The waiting drone asked for a pad every 1 s (`waiting_room` timer), so a freed pad could stay empty for up to 1 s | The base sends `PAD` to the first drone in line at the instant a pad frees |
| Take-off | The drone called `station.take_off()` | The drone sends `TAKE_OFF` |
| Swarm: who is on the base | Each charging drone broadcast its own `ChargingNotificationMessage` every 1 s. Waiting drones were silent | The base broadcasts a `BaseHeartbeatMessage` every 1 s: drones charging **and** waiting |
| Swarm: `N` of the congestion term | Drones believed CHARGING | Drones believed on the base: CHARGING or WAITING |
| Centralized | No communication handler | A handler, used only by the messages with the base |
| expanding_testing | A `recharge_base.py` per system, with a `ChargingRegistry` the drones plugged into | The shared `Analytical/recharge_base.py`. The stations are a plain list, written only by the bases |

## Why

- **The base is a node, so it can grow.** It is the place to add base behaviour
  later, and it is already how `expanding_testing` modelled its bases.
- **The base knows who is on the ground.** It is the natural sender of that
  information. Before, only the charging drones said it, and a waiting drone said
  nothing, so the swarm drones did not know about the queue (open point of
  `waiting_room.md`, "the swarm estimate does not know about waits").
- **No empty pad.** The push from the base removes the up-to-1 s gap of the polling.

## The messages

```
drone lands      ── LANDED (its battery) ──▶  base: a pad free and nobody waiting?
                 ◀── PAD ─────────────────    yes: at once. No: when its turn comes
drone charges TIME_TO_RECHARGE (300 s) on the pad
drone takes off  ── TAKE_OFF ─────────────▶  base: the pad goes to the first drone of
                                              the waiting room, with a PAD, same instant

swarm only, every 1 s while some drone is on the ground:
base ── BASE_HEARTBEAT (pads, charging, waiting) ──▶ every drone in range
```

- gradysim delivers a message with no delay as a new event at the same time. So on a
  free pad the drone is WAITING for 0 s, and a freed pad passes to the next drone at
  the take-off instant.
- A landed drone and its base are at the same point (the base antenna is at the
  drones' 50 m), so these messages are always in range. The medium has no failures.
  With failures, a drone whose `LANDED` or `PAD` was lost would wait forever. That is
  not handled.
- The message types are 100 to 103, apart from the drones' own 0 to 4.

## Where in the code

**`<system>/recharge_base.py`** (one copy per system since 2026-11-25, each with only
what its system uses):
- `BaseMessageType`, and the TypedDicts of the messages. `BASE_HEARTBEAT` and
  `BaseHeartbeatMessage` are only in the swarm copies.
- `base_node_id(number_of_drones, base_index)`: the base is node `n + b`.
- `send_landed` and `send_take_off`: what the drones call.
- `RechargeBase`, the node:
  - `instantiate` makes its pads and its waiting room: `pads`, `charging` (a dict
    `{id: (charge start, battery)}`), `waiting` (first landed first), the waits and the
    landings of the run.
  - `land(id, now, battery)` and `take_off(id, now)`. `take_off` returns the drones that
    get a pad now.
  - `waiting_room_starts(now)` gives when each drone of its waiting room will start
    charging. Swarm copies and centralized only: the swarm base puts it in its
    heartbeat, the centralized scheduler asks the base for it. It needs
    `TIME_TO_RECHARGE`, which the base gets from `main.py` (since 2026-11-25; before,
    the function took the charge time of the drone that called it).
  - `summary` and `report` give the waits of the run to the reports.
  - `handle_packet` answers `LANDED` and `TAKE_OFF` and ignores the drones' broadcasts.
  - Swarm copies: `charging_rows` and `waiting_rows` build the heartbeat, and the
    `base_heartbeat` timer runs with `broadcast=True`.
- `recharge_base_factory(pads, base_index, mode)`. The swarm copies and centralized take
  `time_to_recharge` after `pads`, and the swarm copies `broadcast` too.
- `BASE_ANTENNA_ALTITUDE = 50`.

**Each `main.py`:**
- After the drones, `builder.add_node(recharge_base_factory(Nc, mode=mode),
  (x, y, BASE_ANTENNA_ALTITUDE))`. Swarm adds `broadcast=BROADCAST_BASE` (True, and only
  the swarm `main.py` files have it). `expanding_testing` adds one node per base, with
  `PADS_PER_BASE` pads (Nc by default) and its `base_index`. Swarm and centralized
  also give the base `TIME_TO_RECHARGE`.
- `TIME_TO_RECHARGE` (300 s) is a constant of every `main.py` since 2026-11-25. The
  drones get it as `time_to_recharge` in `drone_protocol_factory`, and the base gets the
  same value where it uses it. Before, each protocol hardcoded 300.
- After the build, the node itself goes to what reads it:
  `global_map.recharge_base = simulation.get_node(base_node_id(n)).protocol_encapsulator.protocol`
  for the reports (`global_map.recharge_bases`, one per base, in `expanding_testing`).
  Centralized also sets `world.base` to it.

**Each protocol:**
- Landing: zero velocity, `battery.shutdown()`, WAITING, `send_landed(...)`.
- `PAD_MESSAGE` while WAITING calls `start_charging()`. It is handled before the status
  gate of `handle_packet`.
- `recharge_done` calls `send_take_off(...)` first.
- Removed: `charging_station` from the config and the factory, `WAITING_CHECK_INTERVAL`,
  the `waiting_room` timer.

## Swarm: the base heartbeat

`Drone.handle_base_heartbeat` (swarm and expanding swarm) replaces the handling of the
charging notification.

- **A drone on a pad** gets the row `(battery, time its charge started, CHARGING)`, as
  the notification gave.
- **A drone of the waiting room** gets `(battery at landing, time its charge will start,
  WAITING)`. The base works that start out and sends it in the heartbeat
  (`RechargeBase.waiting_room_starts`, since 2026-11-25; before, each receiver
  computed it from the landing times in the heartbeat):
  - the pads free at the end of the charges on them;
  - the k-th drone in line takes the k-th pad that frees;
  - each charge takes `TIME_TO_RECHARGE`, 300 s since 2026-11-25 (100 s before).

  This is exact: no drone can overtake the line, and the pad passes at the take-off
  instant. The base is the one that hands the pads over, so the rule of the line is
  written only there, and the receiver stores the start as it gets it.
- **Gossip stays consistent.** The CHARGING row that follows has the same time as the
  WAITING row, so it is not "new". The next stay on the base always has a later time,
  so the newest-row-wins merge keeps the right one.
- **`estimate_drone_battery`, WAITING case.** Before its start, the drone is on the base
  with the battery it landed with. From the start on, it is the CHARGING case.
- **`aproximate_battery_consumption_model` counts the drones on the base**, charging or
  waiting. In `charge_margin` the excess `N + 1 − Nc` is then the waiting room this drone
  would make if it went now. Before the pads, all the drones at the base were CHARGING,
  so this is the same meaning `N` had then.
- **Re-decision.** A MAPPING or GOING_TO_BASE drone chooses its destination again when a
  drone is new on the base, as it did on a new charging notification. WAITING → CHARGING
  is not new.
- **Recorder and estimation metrics.** The "true" number of drones charging and the
  `true_charging` of a target now also count WAITING. WAITING is a target status of its
  own in the error table. `belief_age` is negative for a WAITING row: the time left until
  its charge starts.
- **`main_test.py`.** A decision made on a base heartbeat keeps the trigger name
  `charging_notification`, so the plots of older recordings still read.

## The other systems

| System | Base broadcasts | Notes |
|---|---|---|
| egocentric, naive | no | The drones only act on `PAD` while WAITING |
| centralized | no | `main.py` now adds a `CommunicationHandler` (range `TRANSMISSION_RANGE`) that only carries the messages with the base. The world reads the pads and the waiting room of the base node as the oracle (`world.base`, set by `main.py` after the build; a `ChargingStation` handed by `main.py` until 2026-11-25). `check_centralized.py` now checks that the only messages are with the base |
| expanding swarm | yes, each base about itself (every base about every base until 2026-11-25) | One node per base, each with its own pads and waiting room (they replace `ChargingRegistry`). A drone sends to `base_node_id(n, recharge_base_index)` |
| expanding egocentric, naive | no | as egocentric |

## Since 2026-11-25: the pads inside the node, one copy per system

| | 2026-11-24 | Since 2026-11-25 |
|---|---|---|
| Pads and waiting room | A `ChargingStation` (in `charging_station.py`) made by `main.py`, handed to the node and kept by `main.py` | Attributes of the node, made in `RechargeBase.instantiate`. Only the node writes them |
| Reports (`main_test.py`) | `global_map.charging_station` (`charging_stations`) | `global_map.recharge_base` (`recharge_bases`): the node itself, taken from the simulation after the build |
| Centralized scheduler | `CentralizedWorld(station)` | `CentralizedWorld()`, then `world.base` = the node after the build |
| Several bases | A shared list of every station. Each base wrote its own and broadcast all of them, a "second radio" between the bases | Each base knows only itself. The swarm heartbeat tells about its own base. The drones pass on what they heard of the others with the gossip (newest row wins) |
| Swarm heartbeat | `bases`: a list with one entry per base | `pads`, `charging` and `waiting` of the sending base |
| The file | `Analytical/recharge_base.py`, shared by every system | `<system>/recharge_base.py`, one per system |

**Why.** Each base manages its landings and its pads on its own, as a real base would.
This is the basis for several bases. Each system also runs on its own files, as it
already did with its protocol and its fitness.

**Why the pads are made in `instantiate`.** gradysim creates the protocol of every node
when it builds the simulation, but it calls `initialize` only on the first step, node by
node in id order, the drones before the base. Made in `instantiate`, the pads exist from
the build. So `main.py` can hand the node to its readers before the run, and nothing that
reads the base before the base's own `initialize` finds it half made.

**Checks (2026-11-25).** Two seeded runs per system, 1,500 s, 6 drones, hand-tuned, 200 m,
2 bases in `expanding_testing`, before and after:
- swarm, egocentric, naive, centralized, expanding egocentric and expanding naive: **bit-for-bit
  identical**;
- expanding swarm: **different, as expected**, since a base no longer tells about the
  other one. Mean uncertainty +0.0% and +0.5% in the two runs, waiting 446 → 451 and
  399 → 358 drone·s. It has not been measured over many runs;
- the reports of `main_test.py` (swarm, centralized, expanding naive) read the nodes, and
  `check_centralized.py --only run --duration 1500` passed.

## Checks run on 2026-11-24

These are smoke runs with the hand-tuned individuals, 6 drones, 2,500 s and seed 1.
They check the mechanism; they do not compare the systems.

| System | Landings that waited | Wait, mean / longest | Most charging at once | Deaths | Checks |
|---|---|---|---|---|---|
| swarm | 4 of 10 | 44 s / 78 s | 2 (= `Nc`) | 0 | all ok |
| egocentric | 6 of 11 | 58 s / 102 s | 2 | 0 | all ok |
| naive | 8 of 12 | 50 s / 107 s | 2 | 0 | all ok |
| centralized | 0 of 12 | – | 2 | 0 | all ok |
| expanding swarm, bases (±400, ±400), 1 pad each | 3 of 8 | longest 136 s | 1 per base | 0 | all ok |
| expanding egocentric, same | 6 of 10 | longest 163 s | 1 per base | 0 | all ok |
| expanding naive, same | 7 of 12 | longest 120 s | 1 per base | 0 | all ok |

- **WAITING rows are exact.** In a swarm run (6 drones, 2,500 s, seed 1), every WAITING
  row any drone held had exactly the charge start (float-equal) the base later gave that
  drone. That covered 4 waits, plus 1 start predicted after the end of the run.
- **`check_centralized.py`.** `--only scheduler` passed. `--only run --duration 1500`
  passed too: 5 drones, 10 messages sent (5 `LANDED`, 5 `TAKE_OFF`) and 5 `PAD`
  received, and nothing else. It also checks that the same seed gives the same run and
  that a recorded run equals an unrecorded one.
- `multiple_runs.py` and `expanding_testing/multiple_runs_bases.py` start; they were
  not run in full.

## Consequences and open points

- **Expanding swarm runs from before 2026-11-25 are not comparable** with later ones:
  a base now tells only about itself. The other systems did not change.
- **Runs are not comparable** with runs made earlier on 2026-11-24, or before. There
  is one more node, and adding a node changes the order of the events of an instant
  (see the README). Pads now pass without the 1 s gap. In swarm, `N` now counts the
  waiting drones.
- **Swarm `c` changes meaning a little.** It now acts on the queue, not only on the
  pads in use. The hand-tuned `c` = 0.05 and the a/b/c sweep came before this change.
- **No message loss is handled.** See [The messages](#the-messages).
- **Centralized** has a communication handler now. Its drones still send nothing to
  each other.
- **MARL was not changed.** Its protocol copies still send the charging notification
  and have no pads, as noted in `waiting_room.md`.
