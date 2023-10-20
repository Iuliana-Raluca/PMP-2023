from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Definirea variabilelor aleatoare
model = BayesianNetwork([('C', 'I'),
                         ('C', 'A'),
                         ('I', 'A')])

# Definirea Tabelului CPD pentru fiecare variabilă aleatoare
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], #SA FIE INCENDIU FARA CUTREMUR,
                           evidence=['C'], evidence_card=[2])
cpd_a = TabularCPD(variable='A', variable_card=2,
                        values=[[0.999, 0.5, 0.8, 0.02], #A=0 CU TOATE COMBINATIILE DE I SI C
                                [0.001, 0.95, 0.2, 0.98]], #A=1 CU TOATE COMBINATIILE DE I SI C
                        evidence=['C', 'I'], evidence_card=[2, 2])

# Atașarea tabelului CPD la rețea
model.add_cpds(cpd_c, cpd_i, cpd_a)

assert model.check_model()
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

# Crearea unui motor de inferență pe baza modelului
inference = VariableElimination(model)

# Calcularea probabilității că a avut loc un cutremur,  fiind declanșarea alarmei de incendiu
result = inference.query(variables=['C'], evidence={'A': 1})
print(result)  #preluam C(1)
result = inference.query(variables=['I'], evidence={'A': 0})
print(result)  #PRELUAM I(1)
