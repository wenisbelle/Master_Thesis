"""
Swarm with alpha_variance_modifier 0, 0.5 and 1 (everything else 4 _ 100 1 200 0.25 0.25 0.10 0.30)
against the naive hand-tuned runs, per scenario. Run from src/Analytical in the container:

    MPLBACKEND=Agg python3 tmp_sweep_plots/compare_alpha.py
"""
import os
import sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.getcwd())
import plot_multiple_runs as pmr
from multiple_runs import load_runs, time_grid, aligned, run_statistics, cell_distance_to_base

SCENARIOS = [(200, 6), (100, 3), (100, 12), (200, 12)]
##### Each system on its own figures, its α=0 is the reference of the "vs_alpha0" one #####
FAMILIES = {
    "swarm":      {"a00": "swarm α=0", "a05": "swarm α=0.5", "a10": "swarm α=1"},
    "egocentric": {"a00": "egocentric α=0", "a05": "egocentric α=0.5"},
}
##### Okabe-Ito, not the naive pink nor each other within a figure #####
pmr.SYSTEM_STYLES.update({
    "swarm α=0":        {"color": "#0072B2", "ls": "-",          "marker": "o"},
    "swarm α=0.5":      {"color": "#009E73", "ls": "-.",         "marker": "^"},
    "swarm α=1":        {"color": "#E69F00", "ls": (0, (6, 2)),  "marker": "v"},
    "egocentric α=0":   {"color": "#D55E00", "ls": "--",         "marker": "s"},
    "egocentric α=0.5": {"color": "#56B4E9", "ls": ":",          "marker": "D"},
})
STEADY_STATE = 2000.0
WINDOW = 1000.0
FAR = 1000.0  # m, "far cells" are at least this far from the nearest base
OUT = "tmp_sweep_plots"


def load(directory):
    config, runs = load_runs(directory)
    grid = time_grid(config)
    return {"config": config, "grid": grid, "runs": runs,
            "series": [aligned(record, grid) for record in runs],
            "statistics": [run_statistics(record, grid) for record in runs]}


def steady_mean(data, rows):
    steady = data["grid"] >= STEADY_STATE
    return np.array([data["series"][row]["global_uncertainty"][steady].mean() for row in rows])


def far_revisit(data):
    """Mean time between two visits over every revisit of the cells at least FAR from the base."""
    count = total = 0.0
    for record in data["runs"]:
        distance = cell_distance_to_base(record["revisit"]["count"].shape, record["base_positions"])
        far = distance >= FAR
        count += record["revisit"]["count"][far].sum()
        total += record["revisit"]["total"][far].sum()
    return total / count if count else float("nan")


def compare(data, reference):
    """Paired relative difference of the steady mean uncertainty, % of reference, interval and p."""
    pairs = pmr.paired_runs(data, reference)
    rows, reference_rows = zip(*pairs)
    values, reference_values = steady_mean(data, rows), steady_mean(reference, reference_rows)
    mean, low, high = pmr.bootstrap_relative_difference(values, reference_values, True)
    p = float(wilcoxon(values, reference_values).pvalue)
    return float(mean), float(low), float(high), p, len(rows)


def p_text(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def row(label, data, naive, alpha0):
    stats = data["statistics"]
    vs_naive = vs_zero = ""
    n = len(data["runs"])
    if data is not naive:
        m, lo, hi, p, n = compare(data, naive)
        vs_naive = f"{m:+5.1f}% [{lo:+5.1f},{hi:+5.1f}] {p_text(p)}"
    if alpha0 is not None and data is not alpha0 and data is not naive:
        m0, lo0, hi0, p0, _ = compare(data, alpha0)
        vs_zero = f"{m0:+5.1f}% [{lo0:+5.1f},{hi0:+5.1f}] {p_text(p0)}"
    print(f"{label:17s} {n:4d}  {vs_naive:>26s}  {vs_zero:>26s}  "
          f"{np.mean([s['above_nc'] for s in stats]):6.2f}  {np.mean([s['dead'] for s in stats]):4.2f}  "
          f"{np.mean([s['mean_revisit'] for s in stats]):9.0f}  {far_revisit(data):11.0f}")


for transmission_range, drones in SCENARIOS:
    tag = f"{transmission_range}_{drones}"
    naive = load(f"naive/tmp_sweep/a00_{tag}")
    print(f"\n### {drones} drones, {transmission_range:g} m "
          f"(mean uncertainty from {STEADY_STATE:g} s, paired, 95% bootstrap interval)")
    print(f"{'':17s} {'runs':>4s}  {'vs naive':>26s}  {'vs own α=0':>26s}  {'>Nc %':>6s}  {'dead':>4s}  "
          f"{'revisit s':>9s}  {'far ≥1 km s':>11s}")
    row("naive", naive, naive, None)
    for system, alphas in FAMILIES.items():
        variants = {}
        for key, label in alphas.items():
            directory = f"{system}/tmp_sweep/{key}_{tag}"
            if os.path.exists(os.path.join(directory, "config.json")):
                data = load(directory)
                ##### Still running: no run written yet #####
                if data["runs"]:
                    variants[label] = data
        if not variants:
            continue
        alpha0 = variants.get(alphas["a00"])
        for label, data in variants.items():
            row(label, data, naive, alpha0)

        ##### Figures: difference to naive, difference to α=0, revisit by distance #####
        loaded = {"naive": naive, **variants}
        figures = {"vs_naive": pmr.plot_uncertainty_difference(loaded, "naive", WINDOW, STEADY_STATE),
                   "revisit": pmr.plot_revisit(loaded, pmr.DISTANCE_BIN)}
        if alpha0 is not None and len(variants) > 1:
            figures["vs_alpha0"] = pmr.plot_uncertainty_difference(variants, alphas["a00"], WINDOW, STEADY_STATE)
        for name, fig in figures.items():
            if fig is not None:
                fig.savefig(os.path.join(OUT, f"alpha_{system}_{tag}_{name}.png"), bbox_inches="tight", dpi=110)
        pmr.plt.close("all")
