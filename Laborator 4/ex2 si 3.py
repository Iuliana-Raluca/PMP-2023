import numpy as np

lambda_param = 20
mean_time = 2
std_dev_time = 0.5
target_time = 15
target_probability = 0.95

num_experiments = 1000


# Funcție pentru a calcula timpul total de servire pentru un experiment dat
def calculate_total_time(alpha_guess):
    num_clients = np.random.poisson(lambda_param)
    total_time = np.random.normal(loc=mean_time, scale=std_dev_time, size=num_clients).sum()
    total_time += np.random.exponential(scale=alpha_guess)
    return total_time


# Cautare binara pentru a gasi alpha maxim
alpha_lower_bound = 0
alpha_upper_bound = 10  # Presupunem un interval inițial

while alpha_lower_bound < alpha_upper_bound:
    alpha_guess = (alpha_lower_bound + alpha_upper_bound) / 2

    successful_experiments = 0

    # Rulăm experimentele și numărăm câte dintre acestea îndeplinesc condiția dată
    for _ in range(num_experiments):
        total_time = calculate_total_time(alpha_guess)
        if total_time < target_time:
            successful_experiments += 1

    # Calculăm probabilitatea că timpul total de servire este sub 15 minute
    prob_servable = successful_experiments / num_experiments

    # Verificăm dacă probabilitatea este mai mare sau egală cu 95%
    if prob_servable >= target_probability:
        alpha_lower_bound = alpha_guess
    else:
        alpha_upper_bound = alpha_guess


        # Calculăm timpul mediu de așteptare pentru a fi servit al unui client
total_waiting_time = 0
alpha_max = alpha_lower_bound

for _ in range(num_experiments):
    total_time_with_waiting = calculate_total_time(alpha_max)
    # Calculăm timpul de așteptare adăugând timpul de plasare și plată
    waiting_time = total_time_with_waiting - alpha_max
    total_waiting_time += waiting_time

# Calculăm timpul mediu de așteptare
average_waiting_time = total_waiting_time / num_experiments

print(f"alfa maxim pentru a servi clienții într-un timp sub 15 minute cu o probabilitate de 95%: {alpha_max:.2f}")

print(f"Timpul mediu de așteptare pentru a fi servit al unui client: {average_waiting_time:.2f} minute")

