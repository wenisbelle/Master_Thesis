"""
Plots the runs multiple_runs_bases.py wrote, with the four figures of
Analytical/plot_multiple_runs.py, one set for every scenario and range, the systems
(swarm, egocentric, naive) on the same figures. The systems are drawn and labelled as
there, under the name of the system of Analytical they extend: swarm, egocentric and
naive have the same colour, line and marker in both scripts.

- Global uncertainty: at every time, the mean over the runs (line) and one standard
  deviation around it (band).
- Global uncertainty relative to a reference system (--reference, naive by default), in
  %, run k of each system against run k of the reference (same bases and wind): (a) in
  time, in windows of --uncertainty-window seconds; (b) the mean of each run from
  --steady-state on, a dot per run, with a paired Wilcoxon signed-rank test.
- Time on the base, over all the bases together: (a) when, the probability of a drone
  being in the waiting room in windows of --window seconds, (b) how much a drone waits,
  the share of every run, with the paired
  Wilcoxon signed-rank test when exactly two systems are drawn (run k has the same bases
  and wind on every system), (c) how many at once, the occupancy of the bases.
- Time between two visits of a cell by the distance from the cell to its nearest base,
  in bins of --distance-bin metres: the mean (line) and one std deviation around it
  (band), over every revisit of every run.
See Analytical/plot_multiple_runs.py for how each panel is computed.

    cd src
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py                     # 100 m, every scenario run
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --range 500 --cells 200 --drones 20 --bases 3
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --bases 2 3         # every map and swarm size, 2 and 3 bases
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --hand-tuned        # the runs of --hand-tuned
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --system swarm naive --window 100
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --distance-bin 200
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --reference egocentric --steady-state 2000
    python3 Analytical/expanding_testing/plot_multiple_runs_bases.py --range 100 200 500 750 1000 2000 \
        --save figures/bases.pdf --no-show

It reads <system>/<logs-root>/<range>/<scenario>, here in expanding_testing, the
directories multiple_runs_bases.py writes. The scenarios are the directories found
there, <cells>x<cells>_<drones>drones_<bases>bases, kept when they match --cells,
--drones and --bases (every value when a flag is not given). A system without runs
there is left out, a scenario and range without runs of any system is skipped. The
scenarios are not drawn together: the 200 cell map has four times the cells of the 100
cell one, so its global uncertainty, a sum over the cells, is not on the same scale.

With --save NAME.EXT the figures are saved as NAME_<range>_<scenario>_<figure>.EXT, the
figure being uncertainty, uncertainty_difference, charging or revisit, as the --save of
Analytical/plot_multiple_runs.py. The extension sets the format (pdf, png, ...).
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt

EXPANDING_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICAL_DIR = os.path.dirname(EXPANDING_DIR)
if ANALYTICAL_DIR not in sys.path:
    sys.path.insert(0, ANALYTICAL_DIR)

from multiple_runs import TRANSMISSION_RANGE, system_dir
from plot_multiple_runs import (WINDOW, DISTANCE_BIN, UNCERTAINTY_WINDOW, REFERENCE_SYSTEM, load_systems,
                                plot_uncertainty, plot_uncertainty_difference, plot_charging, plot_revisit)
from multiple_runs_bases import (SYSTEMS, DEFAULT_LOGS_ROOT, HAND_TUNED_LOGS_ROOT, scenario_label,
                                 parse_scenario, source_system)


def found_scenarios(systems, logs_root: str, transmission_range: float, cells, drones, bases) -> list:
    """
    The scenarios with a directory under <system>/<logs_root>/<range> for any of the systems,
    kept when they match cells, drones and bases (lists, None keeps every value), sorted by
    map size, drones and bases.
    """
    wanted = (cells, drones, bases)
    scenarios = set()
    for system in systems:
        directory = os.path.join(system_dir(system), logs_root, f"{transmission_range:g}")
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            values = parse_scenario(name)
            if values is not None and all(allowed is None or value in allowed
                                          for value, allowed in zip(values, wanted)):
                scenarios.add(name)
    return sorted(scenarios, key=parse_scenario)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Global uncertainty, drones charging above Nc and time between two visits of a cell "
                    "over the runs of multiple_runs_bases.py")
    parser.add_argument("--system", choices=SYSTEMS, nargs="+", default=list(SYSTEMS),
                        help="system(s) to plot, drawn on the same figures (default: all of them)")
    parser.add_argument("--cells", type=int, nargs="+",
                        help="side(s) of the map in cells of the scenarios to plot (default: every one run)")
    parser.add_argument("--drones", type=int, nargs="+",
                        help="number(s) of drones of the scenarios to plot (default: every one run)")
    parser.add_argument("--bases", "--base", type=int, nargs="+", dest="bases",
                        help="number(s) of bases of the scenarios to plot (default: every one run)")
    parser.add_argument("--hand-tuned", "--hand_tuned", action="store_true",
                        help=f"read the runs of multiple_runs_bases.py --hand-tuned, in {HAND_TUNED_LOGS_ROOT}")
    parser.add_argument("--range", type=float, nargs="+", default=[TRANSMISSION_RANGE], dest="ranges",
                        help="communication range(s) of the runs, in m, a pair of figures each "
                             "(default: %(default)s)")
    parser.add_argument("--logs-root",
                        help="directory of the runs, relative to each system directory, as "
                             f"<range>/<scenario> (default: {DEFAULT_LOGS_ROOT}, {HAND_TUNED_LOGS_ROOT} "
                             "with --hand-tuned)")
    parser.add_argument("--window", type=float, default=WINDOW,
                        help="width of the windows the probability of a drone waiting for a pad is "
                             "estimated in, in s (default: %(default)s)")
    parser.add_argument("--distance-bin", type=float, default=DISTANCE_BIN,
                        help="width of the bins of distance to the nearest base the time between two "
                             "visits is grouped in, in m (default: %(default)s)")
    parser.add_argument("--reference", choices=SYSTEMS,
                        help="system the others are compared to in the difference of uncertainty figure "
                             f"(default: {REFERENCE_SYSTEM} when plotted, otherwise the first --system)")
    parser.add_argument("--uncertainty-window", type=float, default=UNCERTAINTY_WINDOW,
                        help="width of the windows the difference of uncertainty is averaged in, in s "
                             "(default: %(default)s)")
    parser.add_argument("--steady-state", type=float, default=0.0,
                        help="the per-run panel of the difference figure only averages from this time on, "
                             "in s, to leave the first sweep of the map out (default: %(default)s)")
    parser.add_argument("--save",
                        help="also save the figures, as <name>_<range>_<scenario>_<figure><ext>, the figure "
                             "being uncertainty, uncertainty_difference, charging or revisit, the extension "
                             "sets the format")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open the figures, only save them (needs --save)")
    args = parser.parse_args()

    if args.no_show and not args.save:
        parser.error("--no-show without --save would draw nothing")
    if args.distance_bin <= 0:
        parser.error("--distance-bin has to be positive")
    if args.uncertainty_window <= 0:
        parser.error("--uncertainty-window has to be positive")
    if args.logs_root is None:
        args.logs_root = HAND_TUNED_LOGS_ROOT if args.hand_tuned else DEFAULT_LOGS_ROOT
    systems = [SYSTEMS[name] for name in dict.fromkeys(args.system)]

    plotted = 0
    for transmission_range in dict.fromkeys(args.ranges):
        for scenario in found_scenarios(systems, args.logs_root, transmission_range,
                                        args.cells, args.drones, args.bases):
            print(f"\n##### {transmission_range:g} m, {scenario} #####")
            logs_directory = os.path.join(args.logs_root, f"{transmission_range:g}", scenario)
            try:
                loaded = load_systems(systems, logs_directory, transmission_range, None)
            except SystemExit:
                print("no runs of any system, skipped")
                continue
            ##### Under the name of the system of Analytical each one extends, so the figures take the #####
            ##### colour, line, marker and legend label plot_multiple_runs.py gives it                  #####
            loaded = {source_system(system): data for system, data in loaded.items()}

            reference = args.reference or (REFERENCE_SYSTEM if REFERENCE_SYSTEM in loaded else next(iter(loaded)))
            if reference not in loaded:
                print(f"--reference {reference} has no runs here, skipped")
                continue
            if args.steady_state >= min(data["config"]["duration"] for data in loaded.values()):
                print(f"--steady-state {args.steady_state:g} is not shorter than the runs, skipped")
                continue
            plotted += 1

            durations = "/".join(str(d) for d in sorted({data["config"]["duration"] for data in loaded.values()}))
            label = f"{scenario_label(scenario)}, {transmission_range:g} m, {durations} s"
            figures = {"uncertainty": plot_uncertainty(loaded),
                       "uncertainty_difference": plot_uncertainty_difference(loaded, reference,
                                                                             args.uncertainty_window,
                                                                             args.steady_state),
                       "charging": plot_charging(loaded, args.window),
                       "revisit": plot_revisit(loaded, args.distance_bin)}
            ##### None with a single system (difference) or without recorded revisits #####
            figures = {name: fig for name, fig in figures.items() if fig is not None}
            ##### The titles of plot_multiple_runs.py do not know the scenario #####
            titles = {"uncertainty": "Global uncertainty over the runs",
                      "uncertainty_difference": f"Global uncertainty relative to {reference}",
                      "charging": "Time on the base over the runs",
                      "revisit": "Time between two visits of a cell by distance to the nearest base"}
            for name, fig in figures.items():
                fig.suptitle(f"{titles[name]}, {label}")

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
        print(f"\nNo runs found in <system>/{args.logs_root}/<range>/<scenario> for these ranges and "
              f"scenarios, run multiple_runs_bases.py first")
        return 1
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
