"""
Plots the runs multiple_runs_bases.py wrote, with the two figures of
Analytical/plot_multiple_runs.py, one pair for every scenario and range, the systems
on the same figures:

- Global uncertainty: at every time, the mean over the runs (line) and one standard
  deviation around it (band).
- More than Nc drones charging, over all the bases together: (a) when, the probability
  in windows of --window seconds, (b) how much, the share of every run, with the paired
  Wilcoxon signed-rank test of swarm against egocentric (run k has the same bases and
  wind on both), (c) how many at once, the occupancy of the bases.
See Analytical/plot_multiple_runs.py for how each panel is computed.

    cd src
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py                     # 200 m, every scenario
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --range 500 --scenario 200x200_20drones_3bases
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --system swarm --window 100
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --range 100 200 500 750 1000 2000 \
        --save figures/bases.pdf --no-show

It reads <system>/<logs-root>/<range>/<scenario>, here in expanding_testing, the
directories multiple_runs_bases.py writes. A system without runs there is left out, a
scenario and range without runs of any system is skipped. The scenarios are not drawn
together: the 200 cell map has four times the cells of the 100 cell one, so its global
uncertainty, a sum over the cells, is not on the same scale.

With --save NAME.EXT the figures are saved as NAME_<range>_<scenario>_uncertainty.EXT
and NAME_<range>_<scenario>_charging.EXT, the extension sets the format (pdf, png, ...).
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt

EXPANDING_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICAL_DIR = os.path.dirname(EXPANDING_DIR)
if ANALYTICAL_DIR not in sys.path:
    sys.path.insert(0, ANALYTICAL_DIR)

from multiple_runs import TRANSMISSION_RANGE
from plot_multiple_runs import WINDOW, load_systems, plot_uncertainty, plot_charging
from multiple_runs_bases import SYSTEMS, SCENARIOS, DEFAULT_LOGS_ROOT, scenario_label


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Global uncertainty and drones charging above Nc over the runs of multiple_runs_bases.py")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", default=list(SYSTEMS),
                        help="system(s) to plot, drawn on the same figures (default: all of them)")
    parser.add_argument("--scenario", choices=SCENARIOS, nargs="+", default=list(SCENARIOS),
                        help="scenario(s) to plot, a pair of figures each (default: all of them)")
    parser.add_argument("--range", type=float, nargs="+", default=[TRANSMISSION_RANGE], dest="ranges",
                        help="communication range(s) of the runs, in m, a pair of figures each "
                             "(default: %(default)s)")
    parser.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT,
                        help="directory of the runs, relative to each system directory, as "
                             "<range>/<scenario> (default: %(default)s)")
    parser.add_argument("--window", type=float, default=WINDOW,
                        help="width of the windows the probability of more than Nc charging is "
                             "estimated in, in s (default: %(default)s)")
    parser.add_argument("--save",
                        help="also save the figures, as <name>_<range>_<scenario>_uncertainty<ext> and "
                             "<name>_<range>_<scenario>_charging<ext>, the extension sets the format")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open the figures, only save them (needs --save)")
    args = parser.parse_args()

    if args.no_show and not args.save:
        parser.error("--no-show without --save would draw nothing")
    systems = [SYSTEMS[name] for name in dict.fromkeys(args.system)]

    plotted = 0
    for transmission_range in dict.fromkeys(args.ranges):
        for scenario in dict.fromkeys(args.scenario):
            print(f"\n##### {transmission_range:g} m, {scenario} #####")
            logs_directory = os.path.join(args.logs_root, f"{transmission_range:g}", scenario)
            try:
                loaded = load_systems(systems, logs_directory, transmission_range, None)
            except SystemExit:
                print("no runs of any system, skipped")
                continue
            plotted += 1

            durations = "/".join(str(d) for d in sorted({data["config"]["duration"] for data in loaded.values()}))
            label = f"{scenario_label(scenario)}, {transmission_range:g} m, {durations} s"
            figures = {"uncertainty": plot_uncertainty(loaded), "charging": plot_charging(loaded, args.window)}
            ##### The titles of plot_multiple_runs.py do not know the scenario #####
            figures["uncertainty"].suptitle(f"Global uncertainty over the runs, {label}")
            figures["charging"].suptitle(f"Drones charging above Nc over the runs, {label}")

            if args.save:
                stem, extension = os.path.splitext(args.save)
                if os.path.dirname(stem):
                    os.makedirs(os.path.dirname(stem), exist_ok=True)
                for name, fig in figures.items():
                    path = f"{stem}_{transmission_range:g}_{scenario}_{name}{extension or '.pdf'}"
                    fig.savefig(path, bbox_inches="tight")
                    print(f"{path}: saved")
            if args.no_show:
                for fig in figures.values():
                    plt.close(fig)

    if not plotted:
        print(f"\nNo runs found in <system>/{args.logs_root}/<range>/<scenario>, "
              f"run multiple_runs_bases.py first")
        return 1
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
