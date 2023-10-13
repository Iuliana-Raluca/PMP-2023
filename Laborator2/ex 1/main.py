import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


lambda1 = 4
lambda2 = 6

num_samples = 10000

samples = []

for _ in range(num_samples):

    if np.random.uniform() < 0.4:
        # Primul mecanic servește clientul
        service_time = np.random.exponential(scale=1 / lambda1)
    else:
        # Al doilea mecanic servește clientul
        service_time = np.random.exponential(scale=1 / (lambda2 * 1.5))


    samples.append(service_time)

    sample= np.array(samples)

    mean_value = np.mean(sample)
    std_deviation = np.std(sample)

print("Media lui X:", mean_value)
print("Deviația standard a lui X:", std_deviation)


plt.hist(sample,bins=100)
title = "Densitatea distribuției lui X"
plt.title(title)
plt.xlabel(' Timpul')
plt.show()