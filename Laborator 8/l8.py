import pandas as pd
import pymc3 as pm
import numpy as np


#ex1

data = pd.read_csv('Prices.csv')

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    # modelul de regresie
    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])

    y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])

    trace = pm.sample(2000, tune=1000)

    #ex2

    #obtinem HDI pentru beta1 și beta2
hdi_beta1 = pm.stats.hpd(trace['beta1'], hdi_prob=0.95)
hdi_beta2 = pm.stats.hpd(trace['beta2'], hdi_prob=0.95)

print("Estimarea HDI pentru beta1:", hdi_beta1)
print("Estimarea HDI pentru beta2:", hdi_beta2)

#ex3

#pentru a evalua utilitatea predictorilor, am analizat distributiile a posteriori ale acestora.
# daca intervalul de HDI nu contine zero, inseamna că predictorul respectiv are o influenta semnificativa asupra pretului.
# am calculat probabilitatea ca parametrii sa fie mai mari sau mai mici decat zero

prob_beta1_positive = np.mean(trace['beta1'] > 0)
prob_beta2_positive = np.mean(trace['beta2'] > 0)

print("Probabilitatea ca beta1 sa fie pozitiv:", prob_beta1_positive)
print("Probabilitatea ca beta2 sa fie pozitiv:", prob_beta2_positive)


#ex4

# simularea a 5000 de extrageri din distributia a posteriori

posterior_samples = pm.sample_posterior_predictive(trace, samples=5000, model=model)

# construirea intervalului de 90% HDI pentru pretul de vanzare asteptat

hdi_price = pm.stats.hpd(posterior_samples['y'], hdi_prob=0.9)

print("Intervalul HDI pentru prețul de vanzare asteptat:", hdi_price)

#ex5

# simularea a 5000 de extrageri din distributia predictiva posterioara
posterior_predictive_samples = pm.sample_posterior_predictive(trace, samples=5000, model=model)

# calcularea intervalului de 90% HDI pentru distributia predictiva posterioara

hdi_prediction = pm.stats.hpd(posterior_predictive_samples['y'], hdi_prob=0.9)

print("Intervalul HDI pentru distributia predictiva posterioara:", hdi_prediction)


#bonus

#pentru a evalua daca faptul ca producatorul este premium afectează pretul adaug  variabila Premium
# in modelul de regresie pentru a vedea imparcul asupra pretului

with pm.Model() as model_premium:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    beta_premium = pm.Normal('beta_premium', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive']) + beta_premium * (data['Premium'] == 'yes')

    y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])

    trace_premium = pm.sample(2000, tune=1000)

    # obtinem HDI pentru parametri

hdi_beta_premium = pm.stats.hpd(trace_premium['beta_premium'], hdi_prob=0.95)

print("Estimarea HDI pentru impactul premiumului asupra pretului:", hdi_beta_premium)

#Conluzie: Daca intervalul HDI pentru beta_premium nu contine zero, inseamna ca exista o diferenta
# semnificativa în preturile premium si non-premium. De exemplu, daca HDI este în principal pozitiv, atunci producatorii
# premium au tendinta de a avea preturi mai mari.
