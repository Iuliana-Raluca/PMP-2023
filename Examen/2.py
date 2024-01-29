import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def posterior_grid_geom(grid_points=50, observed_head=5, tails=9):
    #parametru observed_head care indica la care aruncare apare prima stema.

    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = stats.geom.pmf(observed_head - 1, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h


grid, posterior = posterior_grid_geom(points, h, t)

plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()
grid, posterior = posterior_grid_geom()
print("Grid:", grid)
print("Posterior:", posterior)

#P2.b. Pentru a găsi valoarea lui teta care maximizeaza probabilitatea a posteriori, cautam maximul din distribuția a posteriori.
# Vom utiliza functia argmax din NumPy pentru a gasi indicele valorilor maxime din distributia a posteriori


# apelam functia pentru a obtine grila si distributia a posteriori
grid, posterior = posterior_grid_geom()

# cautarea indicele maxim a distributiei a posteriori
argmax_index = np.argmax(posterior)

theta_max_posterior = grid[argmax_index]

print("Grid:", grid)
print("Posterior:", posterior)


# afisam indicele si valoarea teta corespunzatoare maximului
print("Indicele maxim:", argmax_index)
print("Valoarea lui teta care maximizeaza probabilitatea a posteriori:", grid[argmax_index])

