"""
GA history of one communication range: the minimum cost of each generation and
the uncertainty, dead and charging terms it is made of.

    python3 Analytical/plot_ga_cost.py --range 200
    python3 Analytical/plot_ga_cost.py --ga-logs Analytical/egocentric/ga_logs --range 500
    python3 Analytical/plot_ga_cost.py --range 200 --save ga_cost_200.pdf

It reads <ga-logs>/ga_logbook_<range>.txt, the logbook main.py writes in train
mode. Logbooks written before the min_<term> columns existed only have the
total, the terms are then left out of the plot.
"""

import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ANALYTICAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GA_LOGS = os.path.join(ANALYTICAL_DIR, "swarm", "ga_logs")

##### Columns of the logbook, label and style of each line. Colour, dash   #####
##### pattern and marker all differ, so they can be told apart without     #####
##### relying on colour. Okabe-Ito colours, as in comparing_swarm_egocentric. #####
SERIES = {
    "min":             {"label": "Total (minimum)", "color": "#000000", "ls": "-",  "marker": "o"},
    "min_uncertainty": {"label": "Uncertainty",     "color": "#0072B2", "ls": "--", "marker": "s"},
    "min_dead":        {"label": "Dead drones",     "color": "#D55E00", "ls": "-.", "marker": "^"},
    "min_charging":    {"label": "Charging",        "color": "#009E73", "ls": ":",  "marker": "D"},
}


def logbook_path(ga_logs, transmission_range):
    return os.path.join(ga_logs, f"ga_logbook_{transmission_range:g}.txt")


def load_logbook(path):
    """
    The logbook table is the header line and one line per generation, the
    best individuals written after it end the table at the first blank line.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[0].split()
    columns = {name: [] for name in header}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != len(header):
            break
        for name, value in zip(header, fields):
            columns[name].append(float(value))
    return columns


def plot_ga_cost(ga_logs, transmission_range, save=None):
    path = logbook_path(ga_logs, transmission_range)
    if not os.path.exists(path):
        raise SystemExit(f"{path}: not found")
    log = load_logbook(path)
    missing = [name for name in SERIES if name not in log]
    if missing:
        print(f"{path}: no {', '.join(missing)} columns, left out of the plot")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    generations = log["gen"]
    ##### Not clipped: a term that stays at 0, often the dead one, lies on the x axis #####
    for name, style in SERIES.items():
        if name in log:
            ax.plot(generations, log[name], lw=1.5, ms=4, clip_on=False, **style)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")
    ax.set_xlim(generations[0], generations[-1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False, handlelength=3.5)
    ax.set_title(f"GA cost of the best individual, {transmission_range:g} m")

    if save:
        fig.savefig(save, bbox_inches="tight")
        print(f"{save}: saved")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimum cost and its uncertainty, dead and charging terms per GA generation")
    parser.add_argument("--ga-logs", default=DEFAULT_GA_LOGS,
                        help="directory with the ga_logbook_<range>.txt files (default: %(default)s)")
    parser.add_argument("--range", type=float, required=True, dest="transmission_range",
                        help="communication range whose GA history is plotted, in m")
    parser.add_argument("--save",
                        help="also save the figure to this file, the extension sets the format (pdf, png, ...)")
    args = parser.parse_args()
    plot_ga_cost(args.ga_logs, args.transmission_range, args.save)
