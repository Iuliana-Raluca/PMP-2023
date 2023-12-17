import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

#modificarea sd de la 10 la 100 face distributia pentru beta mult mai dispersata,
# permitand coeficientilor sa ia valori mai variate și să se indeparteze mai mult de zero


az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

#ex 1) b)

order = 5

x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

#  distributia pentru beta cu sd=100
with pm.Model() as model_p_sd_100:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
    idata_p_sd_100 = pm.sample(2000, return_inferencedata=True)

#distributia pentru beta cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_p_sd_array:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
    idata_p_sd_array = pm.sample(2000, return_inferencedata=True)

#  graficul pentru modelul polinomial cu diferite sd pentru beta
x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

# model cu sd=100
α_p_post_sd_100 = idata_p_sd_100.posterior['α'].mean(("chain", "draw")).values
β_p_post_sd_100 = idata_p_sd_100.posterior['β'].mean(("chain", "draw")).values
y_p_post_sd_100 = α_p_post_sd_100 + np.dot(β_p_post_sd_100, x_1s)

# model cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
α_p_post_sd_array = idata_p_sd_array.posterior['α'].mean(("chain", "draw")).values
β_p_post_sd_array = idata_p_sd_array.posterior['β'].mean(("chain", "draw")).values
y_p_post_sd_array = α_p_post_sd_array + np.dot(β_p_post_sd_array, x_1s)

plt.plot(x_1s[0], y_p_post_sd_100, 'C3', label=f'model sd=100')
plt.plot(x_1s[0], y_p_post_sd_array, 'C4', label=f'model sd=np.array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()

plt.show()
