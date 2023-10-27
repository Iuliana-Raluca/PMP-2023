import numpy as np

# Parametrii
lambda_param = 20  # Parametrul distribuției Poisson
mean_time = 2  # Media distribuției normale pentru timpul de plasare și plată (minute)
std_dev_time = 0.5  # Deviația standard a distribuției normale pentru timpul de plasare și plată (minute)
target_time = 15  # Timpul țintă (minute) pentru servirea comenzilor
target_probability = 0.95  # Probabilitatea țintă pentru servirea comenzilor în timpul specificat

# Numărul total de experimente pentru a genera
num_experiments = 100000


# Funcție pentru a calcula timpul total de servire pentru un experiment dat
def calculate_total_time(alpha):
    num_clients = np.random.poisson(lambda_param)
    total_time = np.random.normal(loc=mean_time, scale=std_dev_time, size=num_clients).sum()
    total_time += np.random.exponential(scale=alpha)
    return total_time


# Cautare binara pentru a gasi α maxim
alpha_lower_bound = 0
alpha_upper_bound = 10  # Presupunem un interval inițial pentru α

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

# Afișăm rezultatul
print(f"α maxim pentru a servi clienții într-un timp sub 15 minute cu o probabilitate de 95%: {alpha_lower_bound:.2f}")