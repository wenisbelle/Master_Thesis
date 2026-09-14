import Analytical.main as M
M.SIMULATION_DURATION = 2000
for run in range(3):
    ra = M.create_and_run_simulation(M.BEST_INDIVIDUAL, mode="train")
    d = ra[0]
    print(f"run {run}: cost={M.evaluate_simulation_cost(ra):.4f} "
          f"unvisited={d['unvisited_cells']:.0f} dist={d['total_distance_traveled']:.0f}")
