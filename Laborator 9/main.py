import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Admission.csv')

# definirea modelului

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)

    p = pm.invlogit(beta0 + beta1 * data['GRE'] + beta2 * data['GPA'])

    # distributia a priori pentru admitere
    admitance = pm.Bernoulli('admitance', p, observed=data['Admission'])

    trace = pm.sample(2000, tune=1000)

az.plot_posterior(trace, var_names=['beta0', 'beta1', 'beta2'])
plt.show()

#ex2
# calcularea granitei de decizie
def calculate_decision_boundary(trace, gre, gpa):
    beta0 = trace['beta0'].mean()
    beta1 = trace['beta1'].mean()
    beta2 = trace['beta2'].mean()
    log_odds = beta0 + beta1 * gre + beta2 * gpa
    return 1 / (1 + np.exp(-log_odds))

# calcularea intervalului HDI
def hdi(probabilities, interval=0.94):
    sorted_probs = np.sort(probabilities)
    tail_prob = (1 - interval) / 2
    lower_index = int(tail_prob * len(sorted_probs))
    upper_index = int((1 - tail_prob) * len(sorted_probs))
    return sorted_probs[lower_index], sorted_probs[upper_index]

# calcularea granitei de decizie pe toate combinatiile de scoruri GRE și GPA
gre_values = np.linspace(data['GRE'].min(), data['GRE'].max(), 100)
gpa_values = np.linspace(data['GPA'].min(), data['GPA'].max(), 100)
decision_boundary = np.zeros((len(gre_values), len(gpa_values)))

for i, gre in enumerate(gre_values):
    for j, gpa in enumerate(gpa_values):
        decision_boundary[i, j] = calculate_decision_boundary(trace, gre, gpa)

plt.contourf(gre_values, gpa_values, decision_boundary.T, cmap='RdBu', alpha=0.8)
plt.scatter(data[data['Admission'] == 1]['GRE'], data[data['Admission'] == 1]['GPA'], color='blue', label='Admis')
plt.scatter(data[data['Admission'] == 0]['GRE'], data[data['Admission'] == 0]['GPA'], color='red', label='Respins')

plt.title('Graniță de Decizie și Intervalul HDI')
plt.xlabel('GRE')
plt.ylabel('GPA')
plt.legend()
plt.show()

 #ex3
# calculul probabilitatii pentru studentul specific
gre_student = 550
gpa_student = 3.5
prob_student_admitance = calculate_decision_boundary(trace, gre_student, gpa_student)

# calculul intervalului HDI pentru probabilitate
hdi_prob_values = hdi(trace['admitance'], interval=0.90)

print(f'Probabilitatea pentru admitere a studentului cu GRE {gre_student} și GPA {gpa_student}: {prob_student_admitance:.4f}')
print(f'Intervalul HDI pentru probabilitatea de admitere: [{hdi_prob_values[0]:.4f}, {hdi_prob_values[1]:.4f}]')
