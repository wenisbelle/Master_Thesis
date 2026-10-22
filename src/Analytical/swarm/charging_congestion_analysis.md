Charging congestion analysis - 2026-10-21
=========================================

Setup: BEST_INDIVIDUAL = [1.07, 1.14, 68.06, 15.65, 741.23, 5.94, 1.69, 1.66, 0.27]
CHARGING_CONGESTION_PENALTY = 1000, DEAD_DRONE_PENALTY = 10000, SIMULATION_DURATION = 10000 s.
Swarm sizes 3, 5, 7, 10 (TRAINING_DRONE_COUNTS), 3 seeds each, with an observer that only
reads each drone's state.

Summary
-------
Raising CHARGING_CONGESTION_PENALTY won't remove the congestion. The penalty is already
about 90% of the cost the GA minimizes. Congestion persists because the protocol has no way
to see that the base is busy and no way to avoid it. Whatever the penalty is, no setting of
the genes can reduce it.


Where the cost comes from (average of 3 seeds)
----------------------------------------------

  drones  Nc   0.01*U   dead   charging penalty   charging share
  ------  --   ------   ----   ----------------   --------------
     3     1    166k      0          231k              58%
     5     1     99k      0        1,408k              93%
     7     2     69k      0        1,003k              94%
    10     3     48k      0          849k              95%
  GA sum        383k      0        3,492k              90%

- The GA mostly sees noise. For the same individual, the 5-drone congestion penalty ranged
  from 1.0M to 1.8M across seeds. That spread is about 8 times the whole uncertainty term,
  while the uncertainty term changes by less than 3% between seeds. With 3 runs per
  evaluation, the ranking of individuals is mostly congestion luck, and coverage quality
  barely shows up.

- There is effectively no coordination. The measured congestion was compared with what you
  would get if every drone charged independently, at random times (the "random baseline"):
    3 drones:  0.6x the random baseline (better than random)
    5 drones:  1.13x (about the same)
    7 drones:  1.8x (worse)
   10 drones:  2.2x (worse)


Why the genes can't reduce it
-----------------------------

1. The congestion signal almost never fires.
   aproximate_battery_consumption_model counts as "charging" the other drones whose
   estimated battery is below MIN_CHARGE_MARGIN (0.20). But drones decide to return at a
   battery of 0.43-0.53 on average. A drone at the base is only counted once its estimate
   decays below 0.20, which happens long after its 100 s recharge has ended.
   - When a drone decided to return, the base was already busy (others charging or flying
     there, at least Nc) in 18-52% of cases.
   - The drone's own estimate said "busy" in 0-21% of cases, and in only 0-1% at 7 and
     10 drones.
   - So c_battery_modifier multiplies a value that is almost always zero, and in practice
     it does nothing.

2. The CHARGING status never reaches anyone.
   A charging drone is frozen: its handle_packet ignores messages and its heartbeat timer
   is not rescheduled. In all 12 runs, no drone ever had another drone marked CHARGING in
   its swarm vector.

3. There is no action to avoid the base.
   Once a drone is GOING_TO_BASE, it never re-evaluates. When it arrives, it always
   charges, however many drones are already there.

4. Every drone starts with a full battery at t=0.
   So the first N arrivals all fall within a 60-210 s window around t=800-950 s, while a
   recharge takes 100 s. That first cycle accounts for 20-47% of the congestion.


How to choose the penalty value
-------------------------------
A useful reference is how much one drone-second of mapping is worth. It can be read from
the 0.01*U column:
  - from 3 to 5 drones:  about 3.3 per drone*s
  - from 5 to 7:         about 1.5
  - from 7 to 10:        about 0.7

The current 1000 per drone*s is 300 to 1400 times that, and even the old value of 100 was
30 to 140 times. So the GA would already give up a lot of coverage to avoid congestion, if
it had any way to do so. Raising the penalty further only shrinks the coverage signal
relative to the noise.

Once the protocol can actually avoid the base, a penalty around 10-50 per drone*s should be
enough. That is still clearly more than a drone's mapping time, so waiting would be
preferred over overlapping. If Nc is a real physical limit (only Nc charging pads exist),
the cleaner option is to enforce it in the protocol with a queue. The penalty then stays at
0, and the cost of waiting shows up naturally in U.


What would actually reduce congestion (most impact first)
---------------------------------------------------------
1. Count the right thing.
   Count the other drones whose last known status is GOING_TO_BASE or CHARGING (column 2
   of swarm_battery_status already holds it), treating each one as occupying a slot until
   roughly time seen + flight time + TIME_TO_RECHARGE. Have a charging drone keep
   answering heartbeats, since it has power on the pad.

2. Allow waiting.
   If the base is full when a drone arrives, it can hover nearby or keep mapping close to
   the base until a slot frees up. At an arrival battery of about 0.4 against the 0.10
   dead threshold, there is about 400 s of reserve, enough to wait out several recharges.

3. Stagger the first cycle.
   For example, give each drone a small charge-margin offset based on its ID during the
   first cycle.


Notes
-----
- The diagnostic scripts used for these numbers are in a temporary directory:
    /tmp/claude-1000/-home-wenis-Mestrado-Tese-Dev-Master-Thesis/2152a232-cebc-48d0-a40e-3ad65ee015a9/scratchpad/cost_breakdown.py
    /tmp/claude-1000/-home-wenis-Mestrado-Tese-Dev-Master-Thesis/2152a232-cebc-48d0-a40e-3ad65ee015a9/scratchpad/analyze.py
- The ga_logbook.txt in the repo (best 115k) comes from a run with the old settings
  (penalty 100, cost divided by the swarm size), so it can't be compared with current costs.
