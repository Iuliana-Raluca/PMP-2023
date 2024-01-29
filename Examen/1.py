import pandas as pd
import pymc3 as pm
import arviz as az

# incarcarea setului de date
data = pd.read_csv('BostonHousing.csv')

# definirea modelului în Pymc3
with pm.Model() as model:

    rm = pm.Data('rm', data['rm'])
    crim = pm.Data('crim', data['crim'])
    indus = pm.Data('indus', data['indus'])

    # variabila dependenta
    medv = pm.Data('medv', data['medv'])

    # coeficienți pentru variabilele independente
    beta_rm = pm.Normal('beta_rm', mu=0, sd=1)
    beta_crim = pm.Normal('beta_crim', mu=0, sd=1)
    beta_indus = pm.Normal('beta_indus', mu=0, sd=1)

    alpha = pm.Normal('alpha', mu=0, sd=1)

    # modelul liniar
    mu = alpha + beta_rm * rm + beta_crim * crim + beta_indus * indus

    # distributia normala pentru variabila dependenta
    medv_obs = pm.Normal('medv_obs', mu=mu, sd=1, observed=medv)

with model:
    trace = pm.sample(1000, tune=1000)

    #ex 2. consider ca variabila care influenteaza cel mai mult rezultatul este numarul mediu de camere(rm), deoarece creșterea numărului mediu de camere este
    # asociata cu o crestere semnificativa a valorii locuintelor (medv).( casele cu mai multe camere sunt, în medie, mai valoroase .

# obtinerea estimarilor HDI
hdi_95 = az.hdi(trace, hdi_prob=0.95)

# Afisarea rezultatelor
print("Estimarile HDI ale parametrilor:")
print("Intercept (alpha):", hdi_95['alpha'])
print("Coeficient pentru 'rm' (beta_rm):", hdi_95['beta_rm'])
print("Coeficient pentru 'crim' (beta_crim):", hdi_95['beta_crim'])
print("Coeficient pentru 'indus' (beta_indus):", hdi_95['beta_indus'])

#simulam extrageri din distributia predictiva posterioara
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=1000)

# calcularea valorilor estimate pentru valoarea locuintelor
medv_estimated = post_pred['medv_obs']

# calcularea intervalului de predictie de 50% HDI
hdi_50 = az.hdi(medv_estimated, hdi_prob=0.5)

# afișarea rezultatelor
print("Intervalul de predictie de 50% HDI pentru valoarea locuintelor:")
print(hdi_50)
