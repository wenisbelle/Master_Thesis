import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

class FitnessEvaluator:
    def __init__(self,
                 map_width: int,
                 map_height: int,
                 distance_between_cells: int,
                 camera_angle:float,
                 distance_norm: float,
                 distance_between_drone_norm: float,
                 base_variance: float,
                 alpha_variance_modifier: float,
                 kernel_n_sigma: int = 3,
                 information_decay_rate: float = 0.001,
                 drone_speed: float = 10.0,
                 number_of_cells_x_y: int = 10):

        self.map_width = map_width
        self.map_height = map_height
        self.camera_angle = camera_angle
        self.distance_between_cells = distance_between_cells
        self.distance_norm = distance_norm
        self.distance_between_drone_norm = distance_between_drone_norm
        self.base_variance = base_variance
        self.kernal_region_size = kernel_n_sigma
        self.alpha_variance_modifier = alpha_variance_modifier
        self.NUMBER_OF_CELLS_X_Y = number_of_cells_x_y
        self.INFORMATION_DECAY_RATE = information_decay_rate
        self.DRONE_SPEED = drone_speed

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

    def uncertainty_modified_time_to_travel(self, map_data:np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> np.array:
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
        return map_data * (self.DRONE_SPEED / (dist +10e-5)) + self.INFORMATION_DECAY_RATE

    def drone_exclusion_mask(self, shape: Tuple[int, int], drone_position: Tuple[float, float, float],
                             map_center_offset: float, camera_angle: float = np.pi/3) -> np.ndarray:
        """
        Valid-source mask from one drone's perspective. False for the cell under the drone
        and for every cell whose centre lies within `radius` (physical units) of the drone,
        radius = altitude*tan(camera_angle) to also drop the cells being observed right now.
        """
        rows, cols = shape
        drone_x, drone_y, _ = drone_position
        radius = drone_position[2] * np.tan(camera_angle)
        x_cell, y_cell = self._cell_coords(rows, cols, map_center_offset)
        dist = np.hypot(x_cell - drone_x, y_cell - drone_y)
        valid = dist > radius
        # The nearest cell is always invalid
        valid.flat[np.argmin(dist)] = False
        return valid

    #def energy_effect

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
            if mode == "sum":
                # evaluated with the target cell's own variance, so the correction is exact
                # at borders/holes whenever the local variance is uniform
                full += np.exp(d2 * neg_half_inv_var)
            if abs(di) >= rows or abs(dj) >= cols:
                continue
            # source (i, j) contributes to target (i + di, j + dj)
            si = slice(max(0, -di), rows - max(0, di)); ti = slice(max(0, di), rows + min(0, di))
            sj = slice(max(0, -dj), cols - max(0, dj)); tj = slice(max(0, dj), cols + min(0, dj))
            num[ti, tj] += w[si, sj] * np.exp(d2 * neg_half_inv_var[si, sj])
            k_ref = neg_half_inv_var[ti, tj] if mode == "sum" else neg_half_inv_var[si, sj]
            den[ti, tj] += src[si, sj] * np.exp(d2 * k_ref)
 
        out = np.zeros((rows, cols))
        np.divide(num * full, den, out=out, where=den > 1e-12)
        return out
    
 
    def cells_priority(self, map_data: np.array, drone_position: Tuple[float, float, float], map_center_offset: float) -> list:
        """
        Batches inputs and uses the Interpolator instead of skfuzzy.compute
        """
        rows, cols = map_data.shape
        drone_x, drone_y, _ = drone_position
 
        modified_map_data = self.uncertainty_modified_time_to_travel(map_data, drone_position, map_center_offset)
 
        # Uncertainty on arrival, I^(t + tau), sets the kernel widths (Eq. 11)
        x_cells, y_cells = self._cell_coords(rows, cols, map_center_offset)
        tau = np.hypot(x_cells - drone_x, y_cells - drone_y) / self.DRONE_SPEED
        arrival_uncertainty = map_data + self.INFORMATION_DECAY_RATE * tau
 
        # Remove the drone's own cell as a source, then aggregate (Eq. 8)
        valid = self.drone_exclusion_mask(map_data.shape, drone_position, map_center_offset)
        reward_map = self.gaussian_influence(modified_map_data, arrival_uncertainty, valid)
 
        # Nearest cell centre (cells are indexed by their centres, see _cell_coords)
        current_i = int(np.clip(round((drone_x + map_center_offset)/self.distance_between_cells), 0, rows - 1))
        current_j = int(np.clip(round((drone_y + map_center_offset)/self.distance_between_cells), 0, cols - 1))
 
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
                x_cell = self.distance_between_cells*i - map_center_offset
                y_cell = self.distance_between_cells*j - map_center_offset
                dist = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)
 
                trajectory_cells = self.get_cells_visited_in_trajectory(
                    drone_altitude=drone_position[2],
                    initial_cell=(current_i, current_j),
                    final_cell=(i, j)
                )
                average_trajectory_cells = sum([reward_map[cell[0], cell[1]] for cell in trajectory_cells])/len(trajectory_cells) if trajectory_cells else 0.0
 
                ##### Final fitness #####
                cell_fitness = average_trajectory_cells - dist/self.distance_norm
                fitness_scores.append((cell_fitness, (i, j)))
 
        return fitness_scores

    def both_cells_priority(self, map_data: np.array, first_drone_pos, second_drone_pos, map_center_offset) -> list:
        """
        Fully vectorized fuzzy inference for two drones.
        """
        # Get individual priorities (using the fast method above)
        list1 = self.cells_priority(map_data, first_drone_pos, map_center_offset)
        list2 = self.cells_priority(map_data, second_drone_pos, map_center_offset)

        if not list1 or not list2:
            return []

        # Convert to arrays for vectorization
        p1_vals = np.array([x[0] for x in list1])
        p1_coords = np.array([x[1] for x in list1]) # Shape (N, 2)

        p2_vals = np.array([x[0] for x in list2])
        p2_coords = np.array([x[1] for x in list2]) # Shape (M, 2)

        #Vectorized Sum of Priorities
        # Shape (N, 1) + (1, M) -> (N, M)
        sum_p_matrix = p1_vals[:, np.newaxis] + p2_vals[np.newaxis, :]

        #Vectorized Distance Calculation
        # Convert grid indices to physical coordinates
        phys_p1 = (p1_coords * self.distance_between_cells) - map_center_offset
        phys_p2 = (p2_coords * self.distance_between_cells) - map_center_offset

        # Broadcasting distance: (N, 1, 2) - (1, M, 2)
        diff = phys_p1[:, np.newaxis, :] - phys_p2[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2)) # Shape (N, M)

        # Normalize distance and combine with priorities
        dist_parameter = dist_matrix / self.distance_between_drone_norm

        total_fitness = sum_p_matrix + dist_parameter

        N = p1_coords.shape[0]
        M = p2_coords.shape[0]
        
        # Flatten the fitness matrix to a 1D array
        scores = total_fitness.ravel()
        
        # Repeat p1 coordinates M times for each element (e.g., A, A, A, B, B, B)
        p1_c = np.repeat(p1_coords, M, axis=0)
        
        # Tile p2 coordinates N times (e.g., X, Y, Z, X, Y, Z)
        p2_c = np.tile(p2_coords, (N, 1))
        
        # Combine them quickly without Python-level loops
        # Returns a list of tuples: (fitness_float, array([x1, y1]), array([x2, y2]))
        return list(zip(scores, p1_c, p2_c))
    
    def choose_one_cell(self, fitness_scores: list) -> Tuple[float, float]:
        if not fitness_scores:
            return None
        
        best_cell = max(fitness_scores, key=lambda x: x[0])
        # Return the coordinates
        return [best_cell[1], best_cell[0]]
    
    def choose_two_cells(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        fitness_scores = max(fitness_scores, key=lambda x: x[0])
        best_1 = (fitness_scores[1])
        best_2 = (fitness_scores[2])

        return [[best_1, best_2], fitness_scores[0]]

