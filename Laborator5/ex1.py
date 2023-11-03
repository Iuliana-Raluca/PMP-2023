import pymc3 as pm
import numpy as np
import pandas as pd

data = pd.read_csv('trafic.csv')

with pm.Model() as model:
    alpha = 1.0 / data.mean()
    lambda_1 = pm.Exponential('lambda_1', alpha)  # inainte de ora 7:00
    lambda_2 = pm.Exponential('lambda_2', alpha)  #  intre orele 7:00 și 8:00
    lambda_3 = pm.Exponential('lambda_3', alpha)  # intre orele 8:00 și 16:00
    lambda_4 = pm.Exponential('lambda_4', alpha)  #  intre orele 16:00 și 19:00
    lambda_5 = pm.Exponential('lambda_5', alpha)  #  după ora 19:00

    poisson_1 = pm.Poisson('poisson_1', lambda_1, observed=data['trafic'][0:7 * 60])
    poisson_2 = pm.Poisson('poisson_2', lambda_2,
                           observed=data['trafic'][7 * 60:8 * 60])
    poisson_3 = pm.Poisson('poisson_3', lambda_3,
                           observed=data['trafic'][8 * 60:16 * 60])
    poisson_4 = pm.Poisson('poisson_4', lambda_4,
                           observed=data['trafic'][16 * 60:19 * 60])
    poisson_5 = pm.Poisson('poisson_5', lambda_5, observed=data['trafic'][19 * 60:])

    step = pm.Metropolis()
    trace = pm.sample(10000, step=step)

    pm.summary(trace)