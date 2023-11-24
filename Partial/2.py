import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

#1.

#asiguram ca, de fiecare data cand se ruleaza programul obtinem aceeasi secventa de numere pseudo aleastoare

np.random.seed(42)
timp_mediu_asteptare = np.random.normal(loc=10, scale=2, size=200)

plt.hist(timp_mediu_asteptare, bins=20, density=True, alpha=0.5, color='b')
plt.title('Histograma Timpilor Medii de Așteptare')
plt.xlabel('Timpul mediu de asteptare')
plt.ylabel('Densitate')
plt.show()

#2

#am ales distributia normala deoarece neavand informatii suplimenrae care sa implice a alta alegere, am considera ca e usor de genstionat cand modelam medii .
#alegand o medie de 10 și o deviatie standard de 5,ne asteptam ca timpul mediu de asteptare sa fie in jurul valorii de 10
# am definit modelul Bayesian si am folosit o distributie normala pentru parametrul miu si o distributie normală pentru observarea datelor

with pm.Model() as model:
    # distributia a priori pentru medie
    mu = pm.Normal('mu', mu=10, sigma=5)

    # distributia observatiilor
    observatii = pm.Normal('observatii', mu=mu, sigma=2, observed=timp_mediu_asteptare)

    #Am ales deviatia standard in functie de experienta anterioara si de asteptarile legate de variabilitatea timpilor de asteptare

#3

#am estimat distributia a posteriori pentru parametrul miu. trace contine rezultatele estimarilor
#distributia a posteriori pentru parametrul miu este obtinuta prin esantionarea din spațiul parametrilor
with model:
    trace = pm.sample(1000, tune=1000)
pm.traceplot(trace)
plt.show()

