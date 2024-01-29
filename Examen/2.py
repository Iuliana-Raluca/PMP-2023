import numpy as np
import scipy.stats as stats

def posterior_grid_geom_infinite(grid_points=50, observed_head=5):
    #parametru observed_head care indica la care aruncare apare prima stema.

    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = stats.geom.pmf(observed_head - 1, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid, posterior = posterior_grid_geom_infinite()
print("Grid:", grid)
print("Posterior:", posterior)
