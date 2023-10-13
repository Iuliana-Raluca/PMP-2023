import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

# Parametrii distribuțiilor gamma pentru cei patru servere
alpha = [4, 4, 5, 5]
beta = [3, 2, 2, 3]

# Probabilitățile pentru fiecare server
probabilities = [0.25, 0.25, 0.30, 0.20]

# Calculăm distribuția gamma pentru fiecare server
distributions = [gamma(a, scale=1/b) for a, b in zip(alpha, beta)]

# Funcție pentru calcularea probabilității că X > 3 milisecunde
def calculate_probability(distributions, probabilities, threshold):
    total_probability = 0
    for dist, prob in zip(distributions, probabilities):
        total_probability += prob * (1 - dist.cdf(threshold))
    return total_probability

# Calculăm probabilitatea că X > 3 milisecunde
threshold = 3
probability = calculate_probability(distributions, probabilities, threshold)


print(f"Probabilitatea ca timpul necesar servirii unui client să fie mai mare de 3 milisecunde: {probability:.4f}")

x_values = np.linspace(0, 10, 1000)
y_values = sum(prob * dist.pdf(x_values) for dist, prob in zip(distributions, probabilities))


plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Densitatea distribuției lui X')
plt.xlabel('Timp (milisecunde)')
plt.ylabel('Densitatea de probabilitate')
plt.title('Densitatea distribuției timpului necesar servirii unui client')
plt.legend()
plt.grid(True)
plt.show()