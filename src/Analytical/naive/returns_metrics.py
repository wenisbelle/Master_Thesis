"""
At which battery the drones go back to the base and what happens on the way,
computed from the return events of SimulationRecorder.

Shared by the report of main_test.py (from recorder.returns) and by
plots/plot_returns.py (from the "returns" of the pickle), so the numbers are the
same everywhere. Only numpy, the plots import it without the simulator.

The naive drone has no estimate of the others' battery, so unlike the egocentric
estimation_metrics.py there is no estimation error to measure, only the trips.
"""

import numpy as np


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


def summarize(returns) -> dict:
    all_trips = trips(returns)
    return {
        "trips": {
            "decisions": len(all_trips),
            "outcomes": {outcome: sum(t["outcome"] == outcome for t in all_trips)
                         for outcome in ("arrival", "abort", "death", "open")},
            "battery_at_decision": _stats([t["battery"] for t in all_trips]),
            "battery_at_arrival": _stats([t["battery_end"] for t in all_trips if t["outcome"] == "arrival"]),
            "triggers": {trigger: sum(t["trigger"] == trigger for t in all_trips)
                         for trigger in sorted({t["trigger"] for t in all_trips})},
        },
        "deaths": [{**event, "drone": drone_id} for drone_id in sorted(returns)
                   for event in returns[drone_id] if event["event"] == "death"],
    }


def _fmt_stats(stats, keys=("mean", "std", "min", "max")) -> str:
    if not stats.get("n"):
        return "no samples"
    return ", ".join(f"{key} {stats[key]:.3f}" for key in keys) + f" (n={stats['n']})"


def format_summary(summary) -> str:
    trips_summary = summary["trips"]
    outcomes = ", ".join(f"{count} {name}" for name, count in trips_summary["outcomes"].items() if count)
    lines = [f"Trips to the base: {trips_summary['decisions']} decisions ({outcomes or 'none'})",
             f"  battery when choosing the base: {_fmt_stats(trips_summary['battery_at_decision'])}",
             f"  battery when reaching it:       {_fmt_stats(trips_summary['battery_at_arrival'])}"]
    if trips_summary["triggers"]:
        lines.append("  decided by: " + ", ".join(f"{name} {count}"
                                                   for name, count in trips_summary["triggers"].items()))
    for event in summary["deaths"]:
        lines.append(f"  drone {event['drone']} died at {event['t']:.0f} s while {event['previous_status']}, "
                     f"{event['distance_to_base']:.0f} m from the base")
    return "\n".join(lines)
