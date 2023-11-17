import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
from arviz import hdi

df = pd.read_csv('auto-mpg.csv')
df_cleaned = df.dropna()

#b:

alpha = pm.Normal('alpha', mu=0, tau=0.01)  #valoarea estimata a mpg atunci cand CP este zero
beta = pm.Normal('beta', mu=0, tau=0.01)    #coeficientul de regresie asociat cu CP
sigma = pm.Uniform('sigma', lower=0, upper=10)  #deviatia standard a reziduurilor

@pm.deterministic
def linear_model(x=df_cleaned['horsepower'], alpha=alpha, beta=beta):
    return alpha + beta * x

obs = pm.Normal('obs', mu=linear_model, tau=1.0/sigma**2, observed=True, value=df_cleaned['mpg'])

model = pm.Model([obs, alpha, beta, sigma])
mcmc = pm.MCMC(model)
mcmc.sample(iter=10000, burn=1000)

pm.Matplot.plot(mcmc)
plt.show()


#c:

# obtinem distributiile marginale ale parametrilor

alpha_samples = mcmc.trace('alpha')[:]
beta_samples = mcmc.trace('beta')[:]

# det. dreapta de regresie cel mai bine potrivita

best_line = lambda x: np.mean(alpha_samples) + np.mean(beta_samples) * x


#d:

#afisam dreapta de regresie impreuna cu datele de intrare si regiunea 95% HDI
plt.xlabel('CP')
plt.ylabel('mpg')
plt.title('Dreapta de Regresie și Datele Observate')

x_range = np.linspace(df_cleaned['horsepower'].min(), df_cleaned['horsepower'].max(), 100)
plt.plot(x_range, best_line(x_range), color='red', label='Dreapta de Regresie')
plt.plot(x_range, np.mean(ppc['obs'], axis=0), color='blue', label='Distribuția Predictivă A Posteriori')
plt.fill_between(x_range, hdi[:, 0], hdi[:, 1], color='blue', alpha=0.3, label='95% HDI')

plt.legend()
plt.show()
