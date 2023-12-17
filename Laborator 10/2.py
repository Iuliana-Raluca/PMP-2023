import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

#ex 2)

# crestem numarul de date la 500 de puncte
#ne asiguram ca la fiecare rulare rezultatele generate vor fi aceleasi

np.random.seed(23)


# generam 500 de valori aleatoare uniform distribuite între 0 și 10
# inmultim fiecare valoare cu 10 si apoi sortam rezultatele pentru
# a obtine un șir ordonat de valori pentru variabila x_new

x_new = np.sort(10 * np.random.rand(500))
y_new = 3 * x_new**2 - 2 * x_new + 1 + np.random.normal(0, 5, 500)

order = 5

x_new_p = np.vstack([x_new**i for i in range(1, order+1)])
x_new_s = (x_new_p - x_new_p.mean(axis=1, keepdims=True)) / x_new_p.std(axis=1, keepdims=True)
y_new_s = (y_new - y_new.mean()) / y_new.std()

with pm.Model() as model_p_sd_100_new:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_new_s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_new_s)
    idata_p_sd_100_new = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_p_sd_array_new:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_new_s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_new_s)
    idata_p_sd_array_new = pm.sample(2000, return_inferencedata=True)

x_plot = np.linspace(x_new_s[0].min(), x_new_s[0].max(), 100)

α_p_post_sd_100_new = idata_p_sd_100_new.posterior['α'].mean(("chain", "draw")).values
β_p_post_sd_100_new = idata_p_sd_100_new.posterior['β'].mean(("chain", "draw")).values
y_p_post_sd_100_new = α_p_post_sd_100_new + np.dot(β_p_post_sd_100_new, x_new_s)

α_p_post_sd_array_new = idata_p_sd_array_new.posterior['α'].mean(("chain", "draw")).values
β_p_post_sd_array_new = idata_p_sd_array_new.posterior['β'].mean(("chain", "draw")).values
y_p_post_sd_array_new = α_p_post_sd_array_new + np.dot(β_p_post_sd_array_new, x_new_s)

plt.plot(x_plot, y_p_post_sd_100_new, 'C3', label=f'model sd=100')
plt.plot(x_plot, y_p_post_sd_array_new, 'C4', label=f'model sd=np.array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_new_s[0], y_new_s, c='C0', marker='.', alpha=0.5)
plt.legend()

plt.show()
