import random
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination


def arunca_moneda(probabilitate):
    return random.random() < probabilitate

def simuleaza_joc(numar_simulari):
    castig_jucator0 = 0  #in aceste 2 variabile retinem numarul de castiguri a fiecarui jucator din cele 20000 de simulari
    castig_jucator1 = 0

    for _ in range(numar_simulari):
        # decidem random care jucator incepe cine incepe
        jucator_curent = random.choice([0, 1])

        #prima runda de aruncare
        if jucator_curent == 0:
            stema_jucator0 = arunca_moneda(1/3)
            steme_jucator1 = arunca_moneda(0.5)
            if stema_jucator0 >= steme_jucator1:
                castig_jucator0 += 1
            else:
                castig_jucator1 += 1
        else:
            steme_jucator0 = arunca_moneda(1/3)

        #a doua runda de aruncare

        steme_jucator1 = sum([arunca_moneda(0.5) for _ in range(int(steme_jucator0) + 1)])
        if steme_jucator0 < steme_jucator1:
            castig_jucator1 += 1
        else:
            castig_jucator0 += 1

    procentaj_castig_jucator0 = (castig_jucator0 / numar_simulari) * 100
    procentaj_castig_jucator1 = (castig_jucator1 / numar_simulari) * 100

    print(f"Jucator 0 a castigat {procentaj_castig_jucator0}% dintre jocuri.")
    print(f"Jucator 1 a castigat {procentaj_castig_jucator1}% dintre jocuri.")

# simulam jocul de 20000 ori

simuleaza_joc(20000)


# 2. Definirea rețelei Bayesiane
model = BayesianModel([('coin_player0', 'outcome_player0'),
                       ('coin_player1', 'outcome_player1'),
                       ('outcome_player0', 'winner'),
                       ('outcome_player1', 'winner')])

# estimam parametrii modelului pe baza simularilor

data = []  #definim o lista pentru a stoca datele jocului
for _ in range(20000):  #simulam aruncarile monedei si rezultatele jocului pentru fiecare runda
    coin_player0 = arunca_moneda(1/3)
    coin_player1 = arunca_moneda(0.5)
    outcome_player0 = arunca_moneda(coin_player0)
    outcome_player1 = sum([arunca_moneda(0.5) for _ in range(int(outcome_player0) + 1)])
    winner = 0 if outcome_player0 >= outcome_player1 else 1
    #adaugam un dictionar la lista reata cu rezultatele jocului pentru fiecare iteratie; datele le voi folosi pentru a estima parametrii modelului Bayesian

    data.append({'coin_player0': coin_player0, 'coin_player1': coin_player1,
                 'outcome_player0': outcome_player0, 'outcome_player1': outcome_player1, 'winner': winner})

model.fit(data, estimator=ParameterEstimator)

inference = VariableElimination(model)

# determinam cine a inceput jocul, daca in a doua runda nu s a obtinut nicio stema

evidence = {'outcome_player1': 0}
starting_coin = inference.map_query(variables=['coin_player0'], evidence=evidence)['coin_player0']

print(f"Fata monedei cea mai probabila sa se fi obtinut in prima runda este: {starting_coin}.")


