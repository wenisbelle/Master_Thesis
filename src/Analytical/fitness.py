import numpy as np
from typing import Tuple, List, Optional
import random
import matplotlib.pyplot as plt

class FitnessEvaluator:
    def __init__(self,
                 map_width: int,
                 map_height: int,
                 distance_between_cells: int,
                 camera_angle:float,
                 #distance_norm: float, # tunable
                 base_variance: float,
                 alpha_variance_modifier: float, # tunable
                 energy_gamma: float, # tunable
                 charging_base_multiplier: float, # tunable - K in the article
                 distance_between_drone_norm: float, # tunable - couples the two targets at an encounter
                 kernel_n_sigma: int = 3, # tunable
                 charge_margin: float = 0.25, # fixed
                 discharge_rate: float = 0.001, # fixed, percentage per second. 1000 SECONDS the whole charge is depleted, reaching charge 1.
                 information_decay_rate: float = 0.001,
                 number_of_cells_x_y: int = 10):

        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells
        #self.distance_norm = distance_norm
        self.distance_between_drone_norm = distance_between_drone_norm
        self.base_variance = base_variance
        self.kernal_region_size = kernel_n_sigma
        self.alpha_variance_modifier = alpha_variance_modifier
        self.NUMBER_OF_CELLS_X_Y = number_of_cells_x_y
        self.INFORMATION_DECAY_RATE = information_decay_rate
        self.CHARGE_MARGIN = charge_margin
        self.DISCHARGE_RATE = discharge_rate
        self.GAMMA = energy_gamma
        self.K = charging_base_multiplier

        # Cache of cell centre coordinates, keyed by (rows, cols, map_center_offset)
        self._cell_coords_cache = {}

        # Gaussian aggregation kernel (Zhang et al. 2025, Eqs. 8-11).
        # With the global max in Eq. (11) the ratio is <= 1, so the widest Gaussian has
        # variance base_variance*(1+alpha). The truncation radius is therefore fixed and
        # replaces kernel_region_size. Offsets are in cell units, so base_variance is in cells^2.
        sigma_max = np.sqrt(base_variance * (1.0 + alpha_variance_modifier))
        radius = kernel_n_sigma * sigma_max
        R = int(np.ceil(radius))
        self._kernel_offsets = [
            (di, dj, float(di*di + dj*dj))
            for di in range(-R, R + 1) for dj in range(-R, R + 1)
            if di*di + dj*dj <= radius**2
        ]

    def _cell_coords(self, rows: int, cols: int, map_center_offset: float):
        """
        Physical x/y coordinates of the cell centres, as broadcastable arrays
        of shape (rows, 1) and (1, cols). Cached: they never change between calls.
        """
        key = (rows, cols, map_center_offset)
        coords = self._cell_coords_cache.get(key)
        if coords is None:
            x = np.arange(rows, dtype=np.float64) * self.distance_between_cells - map_center_offset
            y = np.arange(cols, dtype=np.float64) * self.distance_between_cells - map_center_offset
            coords = (x[:, np.newaxis], y[np.newaxis, :])
            self._cell_coords_cache[key] = coords
        return coords


    def position_to_cell(self, position: Tuple[float, float], map_center_offset: float) -> Tuple[int, int]:
        """
        Physical (x, y) in metres -> (row, col) cell indices, clipped to the map.

        Cells are indexed by their centres (see _cell_coords), so this is the exact
        inverse of `index * distance_between_cells - map_center_offset`, which is how
        the protocol turns a returned cell back into a destination. Every value that
        enters the evaluator in metres but leaves it as a target cell goes through here.
        """
        row = round((position[0] + map_center_offset) / self.distance_between_cells)
        col = round((position[1] + map_center_offset) / self.distance_between_cells)
        return (int(np.clip(row, 0, self.map_width - 1)),
                int(np.clip(col, 0, self.map_height - 1)))

    def get_cells_visited_in_trajectory(self, drone_altitude: float, initial_cell: Tuple[int, int], final_cell: Tuple[int, int]) -> list:
        """
        Cells covered by the camera on the way from initial_cell to final_cell.
        The initial cell is NOT included: the drone is already there, and its
        uncertainty is inflated by the near-zero travel time, so counting it
        would bias every trajectory average.
        """
        x0, y0 = initial_cell
        x1, y1 = final_cell

        radius_coverage = drone_altitude * np.tan(self.camera_angle)

        x_min, x_max = min(x0, x1) , max(x0, x1) 
        y_min, y_max = min(y0, y1) , max(y0, y1) 

        X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))

        ### Equation distance between point and line
        A = y1 - y0
        B = x0 - x1
        C = x1 * y0 - y1 * x0
        denominator = np.sqrt(A**2 + B**2)

        if denominator == 0:
            # Drone is evaluating its exact current cell
            return []
        
        distances = np.abs(A * X + B * Y + C) / denominator
        map_size_d = distances * self.distance_between_cells

        # Filter cells within the camera radius
        mask = map_size_d <= radius_coverage
        
        # Extract the valid coordinates
        valid_x = X[mask]
        valid_y = Y[mask]
        
        # Filter out coordinates that fall outside the actual map boundaries,
        # and the initial cell itself (see docstring)
        bounds_mask = (valid_x >= 0) & (valid_x < self.map_width) & (valid_y >= 0) & (valid_y < self.map_height)
        bounds_mask &= ~((valid_x == x0) & (valid_y == y0))
        valid_x = valid_x[bounds_mask]
        valid_y = valid_y[bounds_mask]
        
        # Create the final list of absolute coordinates
        cells_within_trajectory = list(zip(valid_x, valid_y))

        return cells_within_trajectory

    def uncertainty_modified_time_to_travel(self, map_data:np.array, drone_position: Tuple[float, float, float],
                                            map_center_offset: float, drone_speed: float) -> np.array:
        """
        Returns the uncertainty modified by the time to travel to each cell in the map.
        U = U(t+theta)/theta, where theta is the time to travel to the specific cell.
        The time to travel is calculated as the distance to the cell divided by the drone's speed.
        """
        rows, cols = map_data.shape
        drone_x, drone_y, _ = drone_position

        x_cell, y_cell = self._cell_coords(rows, cols, map_center_offset)

        # Distance from the drone to every cell at once
        dist = np.hypot(x_cell - drone_x, y_cell - drone_y)

        # (U + k*theta)/theta == U/theta + k, with theta = dist/speed
        tau = dist / drone_speed
        phi = map_data / (tau + 10e-5) + self.INFORMATION_DECAY_RATE

        # Uncertainty at the arrivel
        I_arrival = map_data + self.INFORMATION_DECAY_RATE * tau
        return I_arrival, phi

    def drone_exclusion_mask(self, shape: Tuple[int, int], drone_position: Tuple[float, float, float],
                             map_center_offset: float) -> np.ndarray:
        """
        Valid-source mask from one drone's perspective. False for the cell under the drone
        and for every cell whose centre lies within `radius` (physical units) of the drone,
        radius = altitude*tan(camera_angle) to also drop the cells being observed right now.
        """
        rows, cols = shape
        drone_x, drone_y, _ = drone_position
        radius = drone_position[2] * np.tan(self.camera_angle)
        x_cell, y_cell = self._cell_coords(rows, cols, map_center_offset)
        dist = np.hypot(x_cell - drone_x, y_cell - drone_y)
        valid = dist > radius
        # The nearest cell is always invalid
        valid.flat[np.argmin(dist)] = False
        return valid

    def energy_effect(self, map_data:np.array, drone_position: Tuple[float, float, float],
                      recharge_base_position: Tuple[float, float], map_center_offset: float,
                      current_charge: float, drone_speed: float) -> np.array:
        """
        Returns the uncertainty modified by the time to travel to each cell in the map.
        U = U(t+theta)/theta, where theta is the time to travel to the specific cell.
        The time to travel is calculated as the distance to the cell divided by the drone's speed.
        """
        rows, cols = map_data.shape
        drone_x, drone_y, _ = drone_position
        x_cell, y_cell = self._cell_coords(rows, cols, map_center_offset)

        # Distance from the drone to every cell at once
        dist = np.hypot(x_cell - drone_x, y_cell - drone_y)

        # Distance from each cell to the recharge base
        base_x, base_y = recharge_base_position
        dist_to_base = np.hypot(x_cell - base_x, y_cell - base_y)

        # Estimated charge to fly to each cell and then to the recharge base
        charge_needed = ((dist + dist_to_base) / drone_speed) * self.DISCHARGE_RATE

        # Safe remaining charge (Eq. 13)
        R = current_charge - charge_needed - self.CHARGE_MARGIN

        # Equation 14, energy aware coeficient.
        # np.minimum, not min: R is a (rows, cols) array, one value per cell.
        return np.exp(self.GAMMA*np.minimum(R, 0.0))
        

    def gaussian_influence(self, weights: np.ndarray, uncertainty: np.ndarray,
                           valid: np.ndarray = None, mode: str = "sum") -> np.ndarray:
        """
        GMM reward aggregation of Zhang et al. (2025), Eqs. (8)-(11), with every cell as a POI.
 
        weights:     combined efficiency per cell (phi~ = eps * eta * phi).
        uncertainty: uncertainty on arrival, I^(t + tau); sets each kernel's width (Eq. 11).
        valid:       boolean mask of cells allowed to act as sources. Invalid cells are removed
                     entirely, exactly like cells beyond the map border.
        mode:
          "sum"  -> Eq. (8), multiplied by (kernel mass of an unobstructed cell / kernel mass
                    actually available), both at the target cell's variance. Identical to the
                    paper wherever the whole kernel window is valid; near borders or excluded
                    cells it compensates for the missing neighbours.
        Excluded cells still get an output value, computed from their valid neighbours.

        return: the full and final atraction map. 
        """
        if mode not in ("sum"):
            raise ValueError("mode must be 'sum'")
        rows, cols = weights.shape
        if valid is None:
            valid = np.ones((rows, cols), dtype=bool)
        if not valid.any():
            return np.zeros((rows, cols))
 
        src = valid.astype(np.float64)
        w = np.where(valid, weights, 0.0)
        u = np.where(valid, uncertainty, 0.0)
 
        # Eq. (11), normalised by the global max over valid cells
        var = self.base_variance * (1.0 + self.alpha_variance_modifier * u / (u.max() + 1e-6))
        neg_half_inv_var = -0.5 / var
 
        num = np.zeros((rows, cols))
        den = np.zeros((rows, cols))       # kernel mass actually received from valid sources
        full = np.zeros((rows, cols))      # "sum" only: mass an unobstructed cell would receive
        for di, dj, d2 in self._kernel_offsets:
            full += np.exp(d2 * neg_half_inv_var)
            if abs(di) >= rows or abs(dj) >= cols:
                continue
            # source (i, j) contributes to target (i + di, j + dj)
            si = slice(max(0, -di), rows - max(0, di)); ti = slice(max(0, di), rows + min(0, di))
            sj = slice(max(0, -dj), cols - max(0, dj)); tj = slice(max(0, dj), cols + min(0, dj))
            num[ti, tj] += w[si, sj] * np.exp(d2 * neg_half_inv_var[si, sj])
            k_ref = neg_half_inv_var[ti, tj]
            den[ti, tj] += src[si, sj] * np.exp(d2 * k_ref)
 
        out = np.zeros((rows, cols))
        np.divide(num * full, den, out=out, where=den > 1e-12)
        return out
    
 
    def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float],
                       recharge_base_position: Tuple[float, float], current_charge: float,
                       drone_speed: float, map_center_offset: float) -> list:
        """
        Batches inputs and uses the Interpolator instead of skfuzzy.compute
        """
        rows, cols = map_data.shape
        drone_x, drone_y, _ = drone_position
 
        I_at_arrival, phi = self.uncertainty_modified_time_to_travel(map_data, drone_position,
                                                                    map_center_offset, drone_speed)
 
        # Map modified by the energy
        energy_modified_values = self.energy_effect(map_data, drone_position, recharge_base_position,
                                                    map_center_offset, current_charge, drone_speed)

        weights = phi * energy_modified_values

        # Remove the drone's own cell as a source, then aggregate (Eq. 8)
        valid = self.drone_exclusion_mask(map_data.shape, drone_position, map_center_offset)
        reward_map = self.gaussian_influence(weights, I_at_arrival, valid)
 
        # Nearest cell centre (cells are indexed by their centres, see _cell_coords)
        current_i, current_j = self.position_to_cell((drone_x, drone_y), map_center_offset)
 
        fitness_scores = []
 
        min_x_cell = max(0, current_i - self.NUMBER_OF_CELLS_X_Y//2)
        max_x_cell = min(rows, current_i + self.NUMBER_OF_CELLS_X_Y//2)
        min_y_cell = max(0, current_j - self.NUMBER_OF_CELLS_X_Y//2)
        max_y_cell = min(cols, current_j + self.NUMBER_OF_CELLS_X_Y//2)
       
        for i in range(min_x_cell, max_x_cell):
            for j in range(min_y_cell, max_y_cell):
                if i == current_i and j == current_j:
                    continue  # Skip the current cell
 
                # Distance
                #x_cell = self.distance_between_cells*i - map_center_offset
                #y_cell = self.distance_between_cells*j - map_center_offset
                #dist = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)
 
                trajectory_cells = self.get_cells_visited_in_trajectory(
                    drone_altitude=drone_position[2],
                    initial_cell=(current_i, current_j),
                    final_cell=(i, j)
                )
                average_trajectory_cells = sum([reward_map[cell[0], cell[1]] for cell in trajectory_cells])/len(trajectory_cells) if trajectory_cells else 0.0
 
                ##### Final fitness #####
                #cell_fitness = average_trajectory_cells - dist/self.distance_norm
                cell_fitness = average_trajectory_cells
                fitness_scores.append((cell_fitness, (i, j)))
 
        return fitness_scores

    def both_cells_priority(self, map_data: np.array, first_drone_pos: Tuple[float, float], second_drone_pos: Tuple[float, float],
                            recharge_base_position: Tuple[float, float], first_drone_current_charge: float, 
                            second_drone_current_charge: float, drone_speed: float, map_center_offset
                            ) -> Optional[Tuple[float, Tuple[int, int], Tuple[int, int], float, float]]:
        """
        Fully vectorized fuzzy inference for two drones.

        Returns the best pair only:
            (best_fitness, first_drone_cell, second_drone_cell, first_cell_fitness, second_cell_fitness)
        or None if either drone has no candidate cell.
        """
        # Get individual priorities (using the fast method above)
        list1 = self.cells_priority(map_data, first_drone_pos, recharge_base_position, first_drone_current_charge, drone_speed, map_center_offset)
        list2 = self.cells_priority(map_data, second_drone_pos, recharge_base_position, second_drone_current_charge, drone_speed, map_center_offset)

        if not list1 or not list2:
            return None

        # Convert to arrays for vectorization
        p1_vals = np.array([x[0] for x in list1])
        p1_coords = np.array([x[1] for x in list1]) # Shape (N, 2)

        p2_vals = np.array([x[0] for x in list2])
        p2_coords = np.array([x[1] for x in list2]) # Shape (M, 2)

        #Vectorized Sum of Priorities
        # Shape (N, 1) + (1, M) -> (N, M)
        sum_p_matrix = p1_vals[:, np.newaxis] + p2_vals[np.newaxis, :]

        # Physical separation of every candidate pair, shape (N, M).
        # map_center_offset cancels in the difference, so the cell indices scale
        # straight to metres. Broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diff = (p1_coords[:, np.newaxis, :].astype(np.float64)
                - p2_coords[np.newaxis, :, :].astype(np.float64)) * self.distance_between_cells
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        # The coupling term, and the whole reason this is a pair problem.
        # sum_p_matrix on its own is separable: max(p1[i] + p2[j]) is an identity
        # equal to (argmax p1, argmax p2), so without a term that depends on BOTH
        # indices the joint choice is exactly the two independent maxima and the
        # encounter buys no coordination. Rewarding separation pushes the two
        # drones onto different parts of the map when they meet.
        dist_parameter = dist_matrix / self.distance_between_drone_norm

        total_fitness = sum_p_matrix + dist_parameter

        # Index of the best pair in the (N, M) matrix, without materializing the pair list
        flat_best = np.argmax(total_fitness)
        i, j = np.unravel_index(flat_best, total_fitness.shape)

        # (best_fitness, cell for drone 1, cell for drone 2, fitness of cell 1, fitness of cell 2)
        return (float(total_fitness[i, j]),
                tuple(p1_coords[i]),
                tuple(p2_coords[j]),
                float(p1_vals[i]),
                float(p2_vals[j]))

    def recharge_base_fitness(self, drone_location: Tuple[float, float], recharge_base_position: Tuple[float, float],
                              current_drone_charge: float, drone_speed: float, ) -> float:
        """
        Returns the fitness score of the base location, which is the distance from the base to the center of the map.
        """
        # Distance from each cell to the recharge base.
        # drone_location is the 3D telemetry position, only x/y matter here.
        base_x, base_y = recharge_base_position
        drone_x, drone_y = drone_location[0], drone_location[1]
        dist_to_base = np.hypot(drone_x - base_x, drone_y - base_y)
        R_individual_safe = current_drone_charge - (dist_to_base / drone_speed) * self.DISCHARGE_RATE - self.CHARGE_MARGIN

        ### For now let's pass the swarm term
        
        return self.K * np.exp(-self.GAMMA*R_individual_safe)        

    
    def choose_one_cell(self, map_data: np.array, drone_location: Tuple[float, float, float],
                        recharge_base_position: Tuple[float, float], current_drone_charge: float,
                        drone_speed: float, map_center_offset: float) -> Optional[Tuple[Tuple[float, float], float, str]]:

        fitness_scores = self.cells_priority(map_data, drone_location,
                                             recharge_base_position, current_drone_charge,
                                             drone_speed, map_center_offset)
        if not fitness_scores:
            return None

        best_cell = max(fitness_scores, key=lambda x: x[0])

        # Check if the fitenss of the best cell is greater than the fitness of the recharge base
        recharge_base_fitness = self.recharge_base_fitness(drone_location, recharge_base_position,
                                                          current_drone_charge, drone_speed)

        if best_cell[0] < recharge_base_fitness:
            ##### The base arrives here in metres, but the caller turns whatever is #####
            ##### returned back into metres as if it were a cell index, so it has   #####
            ##### to leave as a cell index like every other destination.            #####
            base_cell = self.position_to_cell(recharge_base_position, map_center_offset)
            return base_cell, recharge_base_fitness, "going_to_base"
        else:
            return best_cell[1], best_cell[0], "mapping"
    
    def choose_two_cells(self, map_data: np.array, first_drone_location: Tuple[float, float, float],
                         second_drone_location: Tuple[float, float, float], recharge_base_position: Tuple[float, float],
                         first_current_drone_charge: float, second_current_drone_charge: float,
                         drone_speed: float, map_center_offset: float) -> Optional[Tuple[Tuple[float, float], float, Tuple[str, str]]]:

        best_pair = self.both_cells_priority(
            map_data=map_data,
            first_drone_pos=first_drone_location,
            second_drone_pos=second_drone_location,
            recharge_base_position=recharge_base_position,
            first_drone_current_charge=first_current_drone_charge,
            second_drone_current_charge=second_current_drone_charge,
            drone_speed=drone_speed,
            map_center_offset=map_center_offset
        )

        if not best_pair:
            return None

        # fitness_scores is already the best pair:
        # (best_fitness, cell_1, cell_2, fitness_1, fitness_2)
        best_fitness, best_1, best_2, best_1_fitness, best_2_fitness = best_pair

        # Now, let's check if each drone should go to the recharge base instead of the best cell
        # Fitness for the recharge base for each drone
        first_recharge_base_fitness = self.recharge_base_fitness(first_drone_location, recharge_base_position,
                                                                 first_current_drone_charge, drone_speed)
        second_recharge_base_fitness = self.recharge_base_fitness(second_drone_location, recharge_base_position,
                                                                  second_current_drone_charge, drone_speed)

        ##### Same as in choose_one_cell: the base leaves as a cell index #####
        base_cell = self.position_to_cell(recharge_base_position, map_center_offset)

        if best_1_fitness < first_recharge_base_fitness:
            destination_1 = base_cell
            drone_1_action = "going_to_base"
        else:
            destination_1 = best_1
            drone_1_action = "mapping"

        if best_2_fitness < second_recharge_base_fitness:
            destination_2 = base_cell
            drone_2_action = "going_to_base"
        else:
            destination_2 = best_2
            drone_2_action = "mapping"

        return [[destination_1, destination_2], best_fitness, [drone_1_action, drone_2_action]]

