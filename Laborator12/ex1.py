import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#ex1

def posterior_grid(grid_points=50, heads=6, tails=9, prior_type='uniform'):

    grid = np.linspace(0, 1, grid_points)

    if prior_type == 'uniform':
        prior = np.repeat(1/grid_points, grid_points)  # prior uniform
    elif prior_type == 'indicator':
        prior = (grid <= 0.5).astype(int)  # prior indicator
    elif prior_type == 'abs_diff':
        prior = np.abs(grid - 0.5)  # prior diferenta absoluta
    else:
        raise ValueError("Tip de prior invalid. Alegeti 'uniform', 'indicator' sau 'abs_diff'.")

    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return grid, posterior

# Exemplu de utilizare cu diverse tipuri de priori

grid1, posterior1 = posterior_grid(prior_type='uniform')
grid2, posterior2 = posterior_grid(prior_type='indicator')
grid3, posterior3 = posterior_grid(prior_type='abs_diff')


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(grid1, posterior1, label='Prior Uniform')
plt.title('Posterior cu Prior Uniform')
plt.xlabel('Probabilitatea Fetei')
plt.ylabel('Probabilitatea Posteriorului')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(grid2, posterior2, label='Prior Indicator')
plt.title('Posterior cu Prior Indicator')
plt.xlabel('Probabilitatea Fetei')
plt.ylabel('Probabilitatea Posteriorului')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(grid3, posterior3, label='Prior Diferenta Absoluta')
plt.title('Posterior cu Prior Diferenta Absoluta')
plt.xlabel('Probabilitatea Fetei')
plt.ylabel('Probabilitatea Posteriorului')
plt.legend()

plt.tight_layout()
plt.show()
