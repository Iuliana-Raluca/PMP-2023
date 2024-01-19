import arviz as az
import matplotlib.pyplot as plt


centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

# inf despre modelul centrat
print("Modelul Centrat:")
print("Numarul de lanturi:", centered_data.posterior.chain.size)
print("Marimea totala a eșantionului generat:", centered_data.posterior.draw.size)

# distributia posteriori pentru modelul centrat
az.plot_posterior(centered_data, round_to=2, hdi_prob=0.95)
plt.show()

# inf despre modelul necentrat
print("\nModelul Necentrat:")
print("Numărul de lanțuri:", non_centered_data.posterior.chain.size)
print("Mărimea totală a eșantionului generat:", non_centered_data.posterior.draw.size)

# distributia a posteriori pentru modelul necentrat
az.plot_posterior(non_centered_data, round_to=2, hdi_prob=0.95)
plt.show()
