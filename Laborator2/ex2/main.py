import random
import matplotlib.pyplot as plt

# Funcție pentru a simula aruncarea monedelor și a returna rezultatul
def arunca_monezi():
    monezi = ['s', 'b']
    rezultat_aruncare = [random.choice(monezi), random.choices(monezi, weights=[0.7, 0.3])[0]]
    return ''.join(rezultat_aruncare)

# Generăm 100 de rezultate independente ale experimentului
rezultate = [arunca_monezi() for _ in range(100)]

# Numărăm aparițiile fiecărei combinații posibile de rezultate
numar_ss = rezultate.count('ss')
numar_sb = rezultate.count('sb')
numar_bs = rezultate.count('bs')
numar_bb = rezultate.count('bb')

# Creăm un grafic cu distribuția rezultatelor
etichete = ['ss', 'sb', 'bs', 'bb']
numere = [numar_ss, numar_sb, numar_bs, numar_bb]

plt.bar(etichete, numere)
plt.xlabel('Combinații rezultate')
plt.ylabel('Număr de apariții')
plt.title('Distribuția rezultatelor în 100 de aruncări de monezi')
plt.show()