# Hard base capacity and waiting room (2026-11-24)

> Since 2026-11-27 a charge lasts `100 + (1 − battery) · 300` s (`charge_duration` in each
> `recharge_base.py`), not a flat `TIME_TO_RECHARGE`. The rest of this note is as written then.

Since 2026-11-24 the recharge base has **`Nc` charging pads**, and at most `Nc` drones
charge at once. A drone that lands while every pad is taken waits on the ground next
to the base, in a **waiting room** (new status `WAITING`), until a pad frees. On the
ground, waiting or charging, the motors are off: **the battery does not change**, so no
drone can die while it waits.

This file explains the change and where it is in the code. The swarm system is the
reference, the other six systems got the same change (see
[The other systems](#the-other-systems)). The line numbers are those of 2026-11-24.

> **Later the same day the base became a node of the simulation**, see
> [`base_node.md`](base_node.md). The pads, the waiting room and the frozen battery
> below still hold. What no longer holds:
> - the drones do not call the `ChargingStation` any more: they send the base node
>   `LANDED` and `TAKE_OFF`, and the base answers `PAD`. Only the base writes the station;
> - a waiting drone does not ask for a pad every second (`waiting_room` timer and
>   `WAITING_CHECK_INTERVAL` are gone): the base hands a freed pad over at that instant;
> - swarm drones no longer send a charging notification: the base broadcasts the drones
>   charging **and waiting**, and the swarm drones count the waiting ones as on the base;
> - "Why a shared Python object and not a gradysim node" below is the reasoning that was
>   reversed;
> - `expanding_testing` has no `ChargingRegistry` any more. Its bases were the shared node
>   of `Analytical/recharge_base.py` with a list of stations, and since 2026-11-25 each
>   system has its own `recharge_base.py` again, one node per base.
>
> **Since 2026-11-25 `charging_station.py` no longer exists**: the pads and the waiting
> room are attributes of the base node itself, and each system has its own
> `recharge_base.py` (see [`base_node.md`](base_node.md)). The links to
> `charging_station.py` below point to the file as it was on 2026-11-24.

## Contents

- [Before and after](#before-and-after)
- [Why](#why)
- [The rules](#the-rules)
- [Where in the code (swarm)](#where-in-the-code-swarm)
- [What did not change in swarm](#what-did-not-change-in-swarm)
- [What changes in the numbers](#what-changes-in-the-numbers)
- [The other systems](#the-other-systems)
- [Checks run on 2026-11-24](#checks-run-on-2026-11-24)
- [Other changes of the same day](#other-changes-of-the-same-day)
- [Consequences and open points](#consequences-and-open-points)

## Before and after

| | Before | Since 2026-11-24 |
|---|---|---|
| Capacity `Nc` | Soft: any number of drones charged at once. Only the cost penalized more than `Nc` | Hard: `Nc` pads, at least 1. A drone that finds them taken waits landed |
| Drone landing on a full base | CHARGING at once, like on an empty base | WAITING, until the first pad that frees (first landed, first served) |
| Battery on the ground | The battery plugin kept billing **hover power** during the 100 s charge (about 0.0009/s, 0.08 per charge) | Frozen: the plugin is stopped at landing and restarted at take-off |
| Death on the ground | Possible in principle (batteries reached 0.068 on the base, unnoticed) | Impossible: the battery does not change |
| Drones CHARGING | Up to the whole swarm | Never more than `Nc` |
| Congestion in the metrics | Time with more than `Nc` charging, and the cost's congestion term | Time spent in the waiting room (`waiting` series); the congestion term is 0 |

**Yes: when more than `Nc` drones are at the base, `Nc` of them are on the pads
charging and the others are in the waiting room, landed and not charging.**

## Why

1. **The comparison was unfair.** In the a/b/c sweep of 2026-11-24 (8 paired runs, 6
   drones, `Nc` = 2), the swarm with a = b = c = 0 (egocentric-like) had the lowest
   uncertainty, 3.5% below the hand-tuned swarm. It got there by overbooking the base:
   more than 2 drones charged 5.3% of the time, up to 6 at once, so its drones were back
   to mapping sooner. The cost charged it +33% for that, but the uncertainty did not see
   it. A system that ignores the capacity looked best on uncertainty. With hard pads,
   waiting costs mapping time for every system, so uncertainty alone is a fair measure.
2. **No death while waiting.** A drone that waits must not lose battery, otherwise a long
   queue could kill it on the base. On the ground the motors are off, so the plugin is
   stopped.
3. **It fixes an old artefact.** The battery used to drain at hover power during the
   whole charge (`estimation_decision.md`, open issue 1). The swarm estimate assumes a
   constant battery at the base, which is now exactly true.

## The rules

```
                              lands on the base
MAPPING ──▶ GOING_TO_BASE ─────────┬───────────────▶ CHARGING ──(300 s)──▶ MAPPING
   ▲                               │ pad free and         ▲                   (take-off:
   │                               │ nobody waiting       │                    battery full,
   │                               │                      │ first in line      plugin resumed)
   │                               └──▶ WAITING ──────────┘ and a pad is free
   │                                    (asks every 1 s)
   └── (aborted on the way, swarm only)
```

- **Landing.** A GOING_TO_BASE drone is on the base once within 5 cm and below 0.1 m/s
  (unchanged). It then zeroes its velocity, **stops its battery plugin**, and asks the
  station for a pad.
- **A free pad.** If fewer than `Nc` drones are charging **and nobody is waiting**, it
  takes a pad and is CHARGING at once. The "nobody waiting" part keeps the order: a drone
  that just landed cannot overtake one already in the waiting room.
- **The waiting room.** Otherwise it is WAITING. Every second (`WAITING_CHECK_INTERVAL`)
  it asks again. It gets a pad when one is free and it is first in line, so the pads go
  in the order the drones landed. A freed pad can stay empty up to 1 s before the next
  drone takes it.
- **Charging.** `TIME_TO_RECHARGE` on the pad, whatever the battery: 300 s since 2026-11-25
  (100 s, unchanged, when the pads were added).
  The swarm drone broadcasts its charging notification every second, as before, with the
  time **its charge started** (not the time it landed) and its battery then.
- **Take-off.** At the end of the charge the drone frees its pad, gets a full battery,
  **restarts its battery plugin**, and is MAPPING again.
- **Pads.** `max(1, int(drones / 3))`: the same `Nc` as before, at least one pad.

What a drone does in each status:

| | MAPPING / GOING_TO_BASE | WAITING | CHARGING | DEAD |
|---|---|---|---|---|
| Camera, mobility, heartbeat, battery check | yes | no | no | no |
| Answers heartbeats, takes part in encounters | yes | no | no | no |
| Battery plugin (consumption) | on | **off** | **off** (was on) | off |
| Sends | heartbeat, gossip | **nothing** | charging notification (swarm) | dead notification |
| Its map's uncertainty keeps growing and is accumulated | yes (camera) | yes | yes | yes |

## Where in the code (swarm)

### 1. The station: `charging_station.py` (new, shared by every system)

[`charging_station.py`](charging_station.py) holds the class `ChargingStation`, the
bookkeeping of one base: which pads are taken and who waits, in order.

| Method | Lines | What |
|---|---|---|
| `__init__(pads)` | [31–39](charging_station.py#L31-L39) | `pads = max(1, int(pads))`, the list of drones charging, the waiting room (a `deque`), a log of the waits |
| `land(drone_id, now)` | [41–52](charging_station.py#L41-L52) | True: the drone takes a pad now. False: it is put at the end of the waiting room |
| `take_pad(drone_id, now)` | [54–61](charging_station.py#L54-L61) | A waiting drone asks: True when a pad is free and it is first in line |
| `take_off(drone_id)` | [63–66](charging_station.py#L63-L66) | The charge ended: the pad is free |
| `summary(end_time)`, `report(...)` | [68](charging_station.py#L68), [85](charging_station.py#L85) | Landings, waits, longest wait, for the reports |

```python
def land(self, drone_id: int, now: float) -> bool:
    self.landings += 1
    self._landed_at[drone_id] = float(now)
    if not self.waiting and len(self.charging) < self.pads:
        self.charging.append(drone_id)
        return True
    self.waiting.append(drone_id)
    return False

def take_pad(self, drone_id: int, now: float) -> bool:
    if self.waiting and self.waiting[0] == drone_id and len(self.charging) < self.pads:
        self.waiting.popleft()
        self.charging.append(drone_id)
        self.waits.append((drone_id, self._landed_at[drone_id], float(now)))
        return True
    return False
```

**Why a shared Python object and not a gradysim node.** It is built once per simulation
and handed to every drone through the protocol factory, the way `results_aggregator` and
the centralized `world` are. A new node would change the order in which gradysim runs the
events of one instant (see the README of `expanding_testing`), and the base needs no radio
to know which of its own pads are taken. The drones do not decide with it: a drone only
learns that it has to wait once it has landed.

**Why the waiting drone asks every second instead of being called.** Each drone only acts
on itself (its own timers, its own status), as everywhere else in the protocols. The price
is at most 1 s of an empty pad.

### 2. The protocol: `swarm/protocol.py`

| Change | Lines |
|---|---|
| New status `WAITING = 4` | [28](swarm/protocol.py#L28) |
| Config default `'charging_station': None` | [165](swarm/protocol.py#L165) |
| `WAITING_CHECK_INTERVAL = 1.0` | [188](swarm/protocol.py#L188) |
| `self.CHARGING_STATION` read from the config (error if missing) | [203–205](swarm/protocol.py#L203-L205) |
| New method `start_charging()` | [681–696](swarm/protocol.py#L681-L696) |
| Uncertainty accumulated while WAITING too | [703–705](swarm/protocol.py#L703-L705) |
| New timer `"waiting_room"` | [720–727](swarm/protocol.py#L720-L727) |
| Take-off: `take_off`, `battery.resume()` | [729–736](swarm/protocol.py#L729-L736) |
| Landing: `battery.shutdown()`, `land()`, CHARGING or WAITING | [785–806](swarm/protocol.py#L785-L806) |
| Factory: new argument `charging_station` | [1015](swarm/protocol.py#L1015), [1055](swarm/protocol.py#L1055) |

**Landing** ([785–806](swarm/protocol.py#L785-L806)). Before, this block set CHARGING,
recorded the charge start, scheduled the notification and `recharge_done`. Now it lands
the drone and lets the station decide:

```python
self.provider.send_mobility_command(SetVelocityMobilityCommand(0.0, 0.0, 0.0))
self.battery.shutdown()

if self.CHARGING_STATION.land(self.provider.get_id(), self.provider.current_time()):
    self.start_charging()
else:
    self.status = DroneStatus.WAITING
    self.provider.schedule_timer("waiting_room", self.provider.current_time() + self.WAITING_CHECK_INTERVAL)
return
```

The `return` skips rescheduling the mobility timer, so the drone is frozen, as it was
while CHARGING.

**`start_charging()`** ([681–696](swarm/protocol.py#L681-L696)). The old charging code
moved into a method, because it now runs from two places: the landing on a free pad, and
the waiting room when a pad frees. It sets CHARGING, `charging_start_time` and
`battery_at_charging_start` (what the charging notification sends), schedules the
notification and `recharge_done` `TIME_TO_RECHARGE` later (300 s since 2026-11-25, 100 s before).

**The waiting room timer** ([720–727](swarm/protocol.py#L720-L727)). It is handled before
the status gate of `handle_timer`, like `recharge_done` and `dead_notification`, because
the drone is not MAPPING or GOING_TO_BASE when it fires:

```python
if timer == "waiting_room":
    if self.status == DroneStatus.WAITING:
        if self.CHARGING_STATION.take_pad(self.provider.get_id(), self.provider.current_time()):
            self.start_charging()
        else:
            self.provider.schedule_timer("waiting_room", self.provider.current_time() + self.WAITING_CHECK_INTERVAL)
    return
```

**Take-off** ([729–736](swarm/protocol.py#L729-L736)). Two lines were added to
`recharge_done`: `self.CHARGING_STATION.take_off(...)` frees the pad, and
`self.battery.resume()` restarts the consumption after `charge_battery_to_full()`. The
rest (MAPPING, a new destination, all the timers started again) is unchanged.

**Why a WAITING drone is frozen without more code.** Everything a flying drone does is
behind two gates that only let MAPPING and GOING_TO_BASE through:
[`handle_timer`, line 756](swarm/protocol.py#L756) (camera, mobility, heartbeat,
traveled distance, battery check) and [`handle_packet`, line 912](swarm/protocol.py#L912)
(heartbeats, gossip, goto messages, dead notifications). A charging notification heard by
a WAITING drone is only stored: the re-decision it triggers is also limited to MAPPING and
GOING_TO_BASE ([906](swarm/protocol.py#L906)). So a WAITING drone neither maps, nor takes
part in encounters, nor can be sent back to mapping by another drone. Every `!=` test on
the status was checked: none lets WAITING through.

### 3. The battery plugin: gradysim `battery_power/plugin.py`

The protocol uses the plugin's existing `shutdown()` and `resume()` (DEAD drones already
used `shutdown()`). One fix was needed in `resume()`
([plugin.py:189–199](../gradys-sim-nextgen/gradysim/protocol/plugin/battery_power/plugin.py#L189-L199)).
The plugin bills the battery from a timer it reschedules every second. `shutdown()` only
sets a flag, so the timer already scheduled still fires once and then stops. If `resume()`
came less than 1 s after `shutdown()`, it scheduled a new timer while the old one was
still pending, and the plugin would run two loops and bill every second twice. A flag,
`_tick_pending` ([87](../gradys-sim-nextgen/gradysim/protocol/plugin/battery_power/plugin.py#L87),
[117](../gradys-sim-nextgen/gradysim/protocol/plugin/battery_power/plugin.py#L117),
[133](../gradys-sim-nextgen/gradysim/protocol/plugin/battery_power/plugin.py#L133)),
records whether a tick is pending, and `resume()` only schedules one when none is:

```python
def resume(self) -> None:
    if not self._active:
        self._active = True
        if not self._tick_pending:
            self._schedule_next_tick()
```

In the protocols the shortest shutdown is a whole charge (300 s since 2026-11-25, 100 s
before), so this never happened.
The fix makes the plugin safe for any use.

### 4. The simulation and the monitor: `swarm/main.py`

| Change | Lines |
|---|---|
| `from ..charging_station import ChargingStation` | [12](swarm/main.py#L12) |
| A new station for every simulation, `int(number_of_drones/3)` pads | [357–359](swarm/main.py#L357-L359) |
| Passed to the factory | [371](swarm/main.py#L371) |
| Kept on the monitor, for the reports: `global_map.charging_station` | [389–391](swarm/main.py#L389-L391) |
| Monitor: `waiting_drones` and `accomulated_waiting_time` | [196–198](swarm/main.py#L196-L198), [232–234](swarm/main.py#L232-L234) |
| Docstring of the monitor: what the pads change in the cost | [173–178](swarm/main.py#L173-L178) |

The station is created inside `create_and_run_simulation`, so it is new for every
simulation, also when the GA or `check_centralized.py` run several in one process.

The cost ([`congestion_penalty_rate`, 138](swarm/main.py#L138) and the sample at
[220–228](swarm/main.py#L220-L228)) was not changed. Its congestion term counts the
drones CHARGING above `Nc`, which the pads now prevent: from 3 drones on, the term is 0.

### 5. The test report: `swarm/main_test.py`

| Change | Lines |
|---|---|
| GOING_TO_BASE → WAITING is an `arrival`, as GOING_TO_BASE → CHARGING | [124–126](swarm/main_test.py#L124-L126) |
| WAITING → CHARGING is not a trip event (like CHARGING → MAPPING) | [129–132](swarm/main_test.py#L129-L132) |
| New section "Base (pads and waiting room)" | [376–379](swarm/main_test.py#L376-L379) |
| New check: never more drones charging than the pads | [387–392](swarm/main_test.py#L387-L392) |

The arrival battery in "Trips to the base" is the battery at landing, which stays the same
during the wait. The new section looks like this:

```
=== Base (pads and waiting room) ===
Pads: 2. Landings: 10, 5 of them on a full base (50%)
Waiting room: 227 drone*s in total, 45 s per wait on average, 77 s at most
Most drones charging at once: 2
```

### 6. Many runs: `multiple_runs.py`

| Change | Lines |
|---|---|
| New series `"waiting": "waiting_drones"` | [98–101](multiple_runs.py#L98-L101) |
| Stored only when the monitor has it (MARL has not) | [295–296](multiple_runs.py#L295-L296) |
| Older runs, without it, still load | [324](multiple_runs.py#L324) |
| Per run: time in the waiting room, drone·s | [336–339](multiple_runs.py#L336-L339) |
| `summary.txt`: a "waiting s" column and its mean ± std | [370](multiple_runs.py#L370), [401](multiple_runs.py#L401) |

### 7. Plots: `swarm/plots/`

- WAITING has its own colour, Okabe-Ito blue `#0072B2`
  ([plot_drones.py:54](swarm/plots/plot_drones.py#L54)), apart from the green of CHARGING.
- It is drawn as a status band in [plot_more_metrics.py:30](swarm/plots/plot_more_metrics.py#L30),
  [plot_returns.py:48](swarm/plots/plot_returns.py#L48) and
  [plot_estimation.py:255](swarm/plots/plot_estimation.py#L255) / [282](swarm/plots/plot_estimation.py#L282).
- The recorder (`recorder.py`) needed no change: it reads the status names from the enum.

## What did not change in swarm

- **The decision to go back.** `fitness.charge_margin`
  ([fitness.py:639–655](swarm/fitness.py#L639-L655)) is the same:
  `margin = max(MIN, charge_margin + (b if δ > 0 else a)·δ − c·max(0, N̂ + 1 − Nc))`.
  `N̂` is still the number of other drones the drone **believes are charging**; drones in
  the waiting room are not counted in it directly.
- **The battery estimate of the others.** `estimate_drone_battery`
  ([434](swarm/protocol.py#L434)) has no WAITING case, and needs none: a waiting drone is
  silent, so WAITING never reaches the table of beliefs. The others keep projecting it from
  the last row they heard, GOING_TO_BASE: charging for `TIME_TO_RECHARGE` (300 s) from its expected arrival,
  then mapping with a full battery. A long wait therefore looks to them like a charge that
  ends too early, until the drone's real charging notifications arrive with the real start.
- **The charging notification** ([660](swarm/protocol.py#L660)), the recharge time (`TIME_TO_RECHARGE`, 300 s since 2026-11-25),
  the arrival tolerances, the encounters.

## What changes in the numbers

- **Drones charging:** never more than `Nc` (from 3 drones on). The "above `Nc`" share in
  the reports and in `plot_multiple_runs.py` is 0 for new runs.
- **Congestion:** now in the time spent in the waiting room (`waiting` series, drone·s).
- **Uncertainty:** now pays for congestion, because a waiting drone does not map. A system
  that sends many drones back together loses mapping time in the queue.
- **Cost:** the congestion term is 0 (from 3 drones on). The cost is the uncertainty term
  plus the dead term.
- **Battery while charging:** flat at the landing value, instead of falling by up to 0.08.

## The other systems

The same change, with these differences. The line numbers are where the main pieces are.

| System | Protocol | Station built in | Differences |
|---|---|---|---|
| egocentric | [protocol.py:28](egocentric/protocol.py#L28) WAITING, [597](egocentric/protocol.py#L597) `start_charging`, [632](egocentric/protocol.py#L632) waiting room, [705](egocentric/protocol.py#L705) landing | [main.py:354](egocentric/main.py#L354) | `start_charging` has no charging notification and no charge start, egocentric never had them |
| naive | [protocol.py:28](naive/protocol.py#L28), [490](naive/protocol.py#L490), [519](naive/protocol.py#L519), [597](naive/protocol.py#L597) | [main.py:407](naive/main.py#L407), with `charging_base_capacity()` | As egocentric. The naive drone still never looks at `Nc` before going |
| centralized | [protocol.py:42](centralized/protocol.py#L42), [470](centralized/protocol.py#L470), [501](centralized/protocol.py#L501), [577](centralized/protocol.py#L577) | [main.py:404](centralized/main.py#L404) | The scheduler also books the drones of the waiting room, see below. The old check "never more than max(1, Nc) charging" became the pad check ([main_test.py:448](centralized/main_test.py#L448)) |
| expanding_testing (3 copies) | `start_charging`, and the station of the drone's base: `self.CHARGING_REGISTRY.stations[self.recharge_base_index]` | `ChargingRegistry(..., pads_per_base)` in each `main.py` | One station per base ([swarm/recharge_base.py:42](expanding_testing/swarm/recharge_base.py#L42)). `PADS_PER_BASE = None` gives every base `Nc` pads ([swarm/main.py:117](expanding_testing/swarm/main.py#L117)). The soft `Nc` over all the bases is unchanged. A drone plugs into the registry only once it has a pad, so the base heartbeats never list a waiting drone |

**Centralized scheduler** ([world.py:351–370](centralized/world.py#L351-L370)). Its booked
charges were the drones CHARGING and GOING_TO_BASE. A drone in the waiting room is neither,
so without a change the scheduler could plan another drone on a pad already promised to it.
`waiting_room_starts` books each waiting drone at the moment a pad frees for it: in the
order they landed, each at the end of the earliest charge on a pad (a small simulation of
the queue with `heapq`). The station reaches the world through the first drone that
registers ([world.py:287](centralized/world.py#L287)).

The `main.py`, `main_test.py` and plot changes are the same as in swarm in every system.
The multi-base reports have a "Pads and waiting rooms of the bases" section, one line per
base, and a check per base.

## Checks run on 2026-11-24

**Smoke runs**, 6 drones, 2,500 s, hand-tuned individuals, one seed. They check the
mechanism, they are not a comparison of the systems.

| System | Landings on a full base | Wait, mean / longest | Most charging at once | Deaths | Checks |
|---|---|---|---|---|---|
| swarm | 5 of 10 | 45 s / 77 s | 2 (= `Nc`) | 0 | all ok |
| egocentric | 7 of 11 | 62 s / 111 s | 2 | 0 | all ok |
| naive | 7 of 11 | 66 s / 104 s | 2 | 0 | all ok |
| centralized | 0 of 12 | – | 2 | 0 | all ok |
| expanding swarm, 2 bases, 1 pad each | 4 of 8 | longest 183 s | 1 per base | 0 | all ok |
| expanding egocentric, 2 bases, 1 pad each | 4 of 8 | longest 327 s | 1 per base | 0 | all ok |
| expanding naive, 2 bases, 1 pad each | 4 of 10 | longest 190 s | 1 per base | 0 | all ok |

Centralized never waited: its scheduler already spaces the returns.

**Battery on the ground** (swarm, 1,500 s, 6 drones, seed 1). Each drone's recorded
battery was exactly constant through WAITING and CHARGING (for example drone 1: 0.1957 for
the 8 samples of its wait and the 10 of its charge), and fell again after take-off at the
normal rate, about 0.0006/s. So the plugin stops and restarts, and does not run twice.

**`check_centralized.py`.** `--only scheduler` and `--only run --duration 1500` passed. The
run check includes "the same seed gives the same run" and "a recorded run is the same as
an unrecorded one", which would fail if a station leaked from one simulation to the next.

## Other changes of the same day

They are not part of the waiting room, but they change the same runs.

- **Wind held 60 s.** The random air speed N(0, 4) m/s, along the heading, now keeps each
  drawn value for 60 s instead of drawing a new one every second. Over a trip it no longer
  averages out, so a constant drain estimate is no longer exact. In gradysim:
  `random_air_speed_hold` ([config.py:73](../gradys-sim-nextgen/gradysim/protocol/plugin/battery_power/config.py#L73)),
  used in `_air_speed_at_current_tick`
  ([plugin.py:139–142](../gradys-sim-nextgen/gradysim/protocol/plugin/battery_power/plugin.py#L139-L142)).
  In swarm: `SHARED_BATTERY_CONFIG` ([protocol.py:115–120](swarm/protocol.py#L115-L120)).
  The same in every Analytical protocol and in MARL.
- **Hand-tuned margins.**

  | System | Value | Where |
  |---|---|---|
  | swarm | `charge_margin` 0.25, floor `MIN_CHARGE_MARGIN` 0.15 | [main_test.py:53](swarm/main_test.py#L53), [main.py:43](swarm/main.py#L43) |
  | egocentric | `charge_margin` 0.25 | [main_test.py:50](egocentric/main_test.py#L50) |
  | centralized | reserve 0.25 (was 0.20) | [main.py:59](centralized/main.py#L59) |
  | naive | return threshold 0.30. It ignores the distance to the base: at 0.25 a drone landed at 0.113 with the base at (800, 800) | [main.py:57](naive/main.py#L57) |

- **Swarm a, b, c** ([main_test.py:50–52](swarm/main_test.py#L50-L52)): a = 0.2, b = 0.3,
  c = 0.05. In the code, **b applies when the drone is below the swarm mean** (it goes back
  earlier, the anticipation term), and **a when it is above** (it delays), see
  [fitness.py:648](swarm/fitness.py#L648).

## Consequences and open points

- **Runs before 2026-11-24 are not comparable** with newer ones: the wind, the pads and
  the battery on the ground all changed that day.
- **The a/b/c sweep of 2026-11-24 ran before the pads.** Its ranking, where all terms off
  had the lowest uncertainty, came from overbooking the base. It should be run again. With
  the pads, c is no longer about safety: a drone sent to a full base waits safely, and only
  loses mapping time.
- **The GA logbooks** were tuned with a soft capacity and the congestion term of the cost,
  which is now 0.
- **Multi-base pads.** Each base gets `Nc` pads by default, so several bases together have
  more pads than `Nc`. Set `PADS_PER_BASE` in the `expanding_testing/*/main.py` files to
  change it.
- **Fewer than 3 drones.** The base still has 1 pad, but the cost's `Nc` is 0, so its
  congestion term charges every charging drone. This was already so before the pads.
- **The swarm estimate does not know about waits**, see
  [What did not change in swarm](#what-did-not-change-in-swarm).
- **MARL was not changed.** Its protocol copies still let any number of drones charge at
  once, so `MARL/*/check_equivalence.py` (bit-for-bit against Analytical swarm) no longer
  matches.
