# The recharge base as a node of the simulation (2026-11-24)

Earlier on 2026-11-24 the base got hard pads and a waiting room
([`waiting_room.md`](waiting_room.md)). They were a shared Python object
(`ChargingStation`) that every drone called directly. Later the same day the base
became a **gradysim node**, in all seven Analytical systems. The drones now talk to it
with messages, and in swarm **the base, not the charging drones, broadcasts who is on
the ground**: the drones charging and the drones in the waiting room.

## Contents

- [Before and after](#before-and-after)
- [Why](#why)
- [The messages](#the-messages)
- [Where in the code](#where-in-the-code)
- [Swarm: the base heartbeat](#swarm-the-base-heartbeat)
- [The other systems](#the-other-systems)
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
drone charges 100 s on the pad
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

**`recharge_base.py` (new, shared by every system):**
- `BaseMessageType`, and the TypedDicts of the four messages.
- `base_node_id(number_of_drones, base_index)`: the base is node `n + b`.
- `send_landed` and `send_take_off`: what the drones call.
- `RechargeBase`, the node:
  - `handle_packet` answers `LANDED` and `TAKE_OFF` and ignores the drones' broadcasts.
  - The `base_heartbeat` timer runs only with `broadcast=True`.
- `recharge_base_factory(stations, base_index, broadcast, mode)`.
- `BASE_ANTENNA_ALTITUDE = 50`.

**`charging_station.py`:**
- `land(id, now, battery)` also keeps the battery.
- `take_off(id, now)` returns the drones that get a pad now. There is no `take_pad` any
  more.
- `charging` is a dict `{id: (charge start, battery)}`.
- `charging_rows`, `waiting_rows` and `as_message` build the heartbeat.
- `waiting_room_starts(...)` gives when each drone of a waiting room will start
  charging. It is shared by the swarm drones and the centralized scheduler.

**Each `main.py`:**
- After the drones, `builder.add_node(recharge_base_factory([charging_station],
  broadcast=BROADCAST_BASE, mode=mode), (x, y, BASE_ANTENNA_ALTITUDE))`.
- `BROADCAST_BASE` is True in swarm and False elsewhere.
- The station is no longer passed to the drones. `global_map.charging_station` is kept
  for the reports.

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
  WAITING)`. The receiver works that start out from the heartbeat
  (`waiting_room_starts`):
  - the pads free at the end of the charges on them;
  - the k-th drone in line takes the k-th pad that frees;
  - each charge takes 100 s.

  This is exact: no drone can overtake the line, and the pad passes at the take-off
  instant. The base sends the landing time, as asked. The receiver keeps the start
  instead, because that is what the estimate needs.
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
| centralized | no | `main.py` now adds a `CommunicationHandler` (range `TRANSMISSION_RANGE`) that only carries the messages with the base. `CentralizedWorld(station)` gets the station from `main.py`, not from the first drone, and reads it as the oracle. `check_centralized.py` now checks that the only messages are with the base |
| expanding swarm | yes, every base about every base | `charging_stations` (one per base) replaces `ChargingRegistry`. A drone sends to `base_node_id(n, recharge_base_index)` |
| expanding egocentric, naive | no | as egocentric |

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
