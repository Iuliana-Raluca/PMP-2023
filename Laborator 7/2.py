import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

df = pd.read_csv('auto-mpg.csv')
df_cleaned = df.dropna()

plt.scatter(df_cleaned['horsepower'], df_cleaned['mpg'])
plt.xlabel('CP ')
plt.ylabel('mpg')
plt.title('Relația dintre CP și mpg')
plt.show()

#b.

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
