"""
How far the battery estimate of the drones is from the truth, and at which battery
the drones go back to the base, computed from a recording of SimulationRecorder.

Shared by the report of main_test.py (from recorder.drone_arrays() and
recorder.returns) and by plots/plot_estimation.py and plots/plot_returns.py (from
the "states" and "returns" of the pickle), so the numbers are the same everywhere.
Only numpy, the plots import it without the simulator.

The estimate only matters while a drone can decide, MAPPING or GOING_TO_BASE
(DECIDING), so every error is measured there. A target the observer already knows
to be DEAD, or that really is DEAD, is left out of the battery error: the protocol
does not use its battery. A dead drone the observer does not know about still
enters the swarm mean, so it shows up in the error of the mean.
"""

import numpy as np

DECIDING = ("MAPPING", "GOING_TO_BASE")
##### True status of the target, for the error split by state #####
TARGET_STATUSES = ("MAPPING", "GOING_TO_BASE", "WAITING", "CHARGING")
##### Difference of the effective margin, with the estimate or the truth, counted as #####
##### "the estimate changed the margin". 0.05 of a battery is ~70 s of flight.          #####
MARGIN_TOLERANCE = 0.05


def has_beliefs(states) -> bool:
    """False for recordings made before the beliefs were recorded, and for egocentric."""
    return "belief_battery" in states[0] and bool(np.isfinite(states[0]["belief_battery"]).any())


def _deciding(states, drone_id, status_values):
    values = [status_values[name] for name in DECIDING if name in status_values]
    return np.isin(states[drone_id]["status"], values)


def pair_samples(states, status_values) -> dict:
    """
    One entry per (sample, observer i, target j) where i decides with its estimate of j:
    i is DECIDING, j is not DEAD and i does not believe it is. Arrays of the same length:
        estimate, true, error (estimate - true), age, believed_charging, true_charging,
        true_status, observer, target, sample (index in the times)
    """
    N = len(states)
    true_battery = np.stack([states[j]["battery"] for j in range(N)], axis=1)   # (T, N)
    true_status = np.stack([states[j]["status"] for j in range(N)], axis=1)     # (T, N)
    dead = status_values.get("DEAD", np.nan)
    ##### "Charging" is on the base: on a pad or in the waiting room (since 2026-11-24), as the #####
    ##### drones count it, see estimate_drone_battery                                         #####
    on_base = [status_values.get("CHARGING", np.nan), status_values.get("WAITING", np.nan)]

    names = ("estimate", "true", "age", "believed_charging", "true_charging",
             "true_status", "observer", "target", "sample")
    columns = {name: [] for name in names}
    for i in range(N):
        estimate = states[i]["belief_battery"]
        valid = (_deciding(states, i, status_values)[:, None]
                 & (true_status != dead)
                 & (states[i]["belief_status"] != dead)
                 & np.isfinite(estimate))
        valid[:, i] = False
        sample, target = np.nonzero(valid)
        columns["estimate"].append(estimate[sample, target])
        columns["true"].append(true_battery[sample, target])
        columns["age"].append(states[i]["belief_age"][sample, target])
        columns["believed_charging"].append(states[i]["belief_charging"][sample, target] > 0.5)
        columns["true_charging"].append(np.isin(true_status[sample, target], on_base))
        columns["true_status"].append(true_status[sample, target])
        columns["observer"].append(np.full(len(sample), i))
        columns["target"].append(target)
        columns["sample"].append(sample)

    pairs = {name: np.concatenate(values) if values else np.array([])
             for name, values in columns.items()}
    pairs["error"] = pairs["estimate"].astype(float) - pairs["true"].astype(float)
    return pairs


def swarm_term_samples(states, status_values) -> dict:
    """
    The swarm terms of every DECIDING sample of every drone, estimated and true,
    concatenated over the drones, plus observer and sample index.
    """
    names = ("mean_battery_estimate", "mean_battery_true",
             "drones_charging_estimate", "drones_charging_true",
             "charge_margin_estimate", "charge_margin_true")
    columns = {name: [] for name in names + ("observer", "sample")}
    for i in range(len(states)):
        if names[0] not in states[i]:
            continue
        sample = np.nonzero(_deciding(states, i, status_values))[0]
        for name in names:
            columns[name].append(states[i][name][sample].astype(float))
        columns["observer"].append(np.full(len(sample), i))
        columns["sample"].append(sample)
    return {name: np.concatenate(values) if values else np.array([])
            for name, values in columns.items()}


def trips(returns) -> list:
    """
    The trips to the base, one per "decision", closed by what came next: "arrival",
    "abort" (sent back to mapping), "death", or "open" when the run ended first.
    Each trip is the decision event plus drone, outcome, t_end and battery_end.
    """
    result = []
    for drone_id in sorted(returns):
        trip = None
        for event in returns[drone_id]:
            if event["event"] == "decision":
                trip = {**event, "drone": drone_id, "outcome": "open",
                        "t_end": np.nan, "battery_end": np.nan}
                result.append(trip)
            elif trip is not None and trip["outcome"] == "open":
                trip["outcome"] = event["event"]
                trip["t_end"] = event["t"]
                trip["battery_end"] = event["battery"]
    return result


def _stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "median": float(np.median(values))}


def _error_stats(error) -> dict:
    error = np.asarray(error, dtype=float)
    error = error[np.isfinite(error)]
    if not len(error):
        return {"n": 0}
    return {"n": len(error), "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())),
            "p90_abs": float(np.percentile(np.abs(error), 90))}


def summarize(states, returns, status_values) -> dict:
    summary = {"has_beliefs": has_beliefs(states)}
    if summary["has_beliefs"]:
        pairs = pair_samples(states, status_values)
        summary["battery"] = _error_stats(pairs["error"])
        summary["battery_by_status"] = {
            name: _error_stats(pairs["error"][pairs["true_status"] == status_values[name]])
            for name in TARGET_STATUSES if name in status_values}
        summary["age"] = _stats(pairs["age"])
        believed, true = pairs["believed_charging"], pairs["true_charging"]
        summary["charging"] = {
            "true": int(true.sum()), "believed": int(believed.sum()),
            "both": int((true & believed).sum()),
            ##### Of the samples where the target really charges, how many the observer gets #####
            "recall": float((true & believed).sum() / true.sum()) if true.any() else np.nan,
            ##### Of the samples where the observer thinks it charges, how many really do #####
            "precision": float((true & believed).sum() / believed.sum()) if believed.any() else np.nan,
        }

        terms = swarm_term_samples(states, status_values)
        count_error = terms["drones_charging_estimate"] - terms["drones_charging_true"]
        margin_error = terms["charge_margin_estimate"] - terms["charge_margin_true"]
        finite_margin = margin_error[np.isfinite(margin_error)]
        summary["mean_battery"] = _error_stats(terms["mean_battery_estimate"] - terms["mean_battery_true"])
        summary["drones_charging"] = {
            "n": len(count_error),
            "exact": float(np.mean(count_error == 0)) if len(count_error) else np.nan,
            "over": float(np.mean(count_error > 0)) if len(count_error) else np.nan,
            "under": float(np.mean(count_error < 0)) if len(count_error) else np.nan,
        }
        summary["charge_margin"] = {
            **_error_stats(finite_margin),
            "changed": float(np.mean(np.abs(finite_margin) > MARGIN_TOLERANCE)) if len(finite_margin) else np.nan,
        }

    all_trips = trips(returns)
    summary["trips"] = {
        "decisions": len(all_trips),
        "outcomes": {outcome: sum(t["outcome"] == outcome for t in all_trips)
                     for outcome in ("arrival", "abort", "death", "open")},
        "battery_at_decision": _stats([t["battery"] for t in all_trips]),
        "battery_at_arrival": _stats([t["battery_end"] for t in all_trips if t["outcome"] == "arrival"]),
        "triggers": {trigger: sum(t["trigger"] == trigger for t in all_trips)
                     for trigger in sorted({t["trigger"] for t in all_trips})},
    }
    summary["deaths"] = [{**event, "drone": drone_id} for drone_id in sorted(returns)
                         for event in returns[drone_id] if event["event"] == "death"]
    return summary


def _fmt_stats(stats, keys=("mean", "std", "min", "max")) -> str:
    if not stats.get("n"):
        return "no samples"
    return ", ".join(f"{key} {stats[key]:.3f}" for key in keys) + f" (n={stats['n']})"


def format_summary(summary) -> str:
    lines = []
    if summary["has_beliefs"]:
        battery = summary["battery"]
        lines.append("Battery of the others, estimate - truth, while the observer decides:")
        lines.append(f"  {'all':<13} {_fmt_stats(battery, ('bias', 'mae', 'rmse', 'p90_abs'))}")
        for name, stats in summary["battery_by_status"].items():
            lines.append(f"  {name.lower():<13} {_fmt_stats(stats, ('bias', 'mae', 'rmse'))}"
                         f"   <- target really {name}")
        age = summary["age"]
        if age.get("n"):
            lines.append(f"  age of the information: median {age['median']:.0f} s, max {age['max']:.0f} s")
        charging = summary["charging"]
        lines.append(f"Target charging: really {charging['true']} samples, believed {charging['believed']}, "
                     f"both {charging['both']} (recall {charging['recall']:.0%}, "
                     f"precision {charging['precision']:.0%})")
        lines.append(f"Swarm mean battery, estimate - truth: "
                     f"{_fmt_stats(summary['mean_battery'], ('bias', 'mae', 'rmse'))}")
        count = summary["drones_charging"]
        if count["n"]:
            lines.append(f"Drones charging, estimate vs truth: exact {count['exact']:.0%}, "
                         f"over {count['over']:.0%}, under {count['under']:.0%} of the samples")
        margin = summary["charge_margin"]
        if margin.get("n"):
            lines.append(f"Effective charge margin, estimate - truth: "
                         f"{_fmt_stats(margin, ('bias', 'mae'))}; differs by more than "
                         f"{MARGIN_TOLERANCE} in {margin['changed']:.0%} of the samples")
    else:
        lines.append("No battery estimate in the recording (egocentric, or recorded before it existed)")

    trips_summary = summary["trips"]
    outcomes = ", ".join(f"{count} {name}" for name, count in trips_summary["outcomes"].items() if count)
    lines.append(f"Trips to the base: {trips_summary['decisions']} decisions ({outcomes or 'none'})")
    lines.append(f"  battery when choosing the base: {_fmt_stats(trips_summary['battery_at_decision'])}")
    lines.append(f"  battery when reaching it:       {_fmt_stats(trips_summary['battery_at_arrival'])}")
    if trips_summary["triggers"]:
        lines.append("  decided by: " + ", ".join(f"{name} {count}"
                                                   for name, count in trips_summary["triggers"].items()))
    for event in summary["deaths"]:
        lines.append(f"  drone {event['drone']} died at {event['t']:.0f} s while {event['previous_status']}, "
                     f"{event['distance_to_base']:.0f} m from the base")
    return "\n".join(lines)
