from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 1.0  # Evolution time
n_trotter = 5  # Number of Trotter steps

# Define the circuit
qc = QuantumCircuit(1)

# Define time evolution for H0, V1, and V2
def unperturbed_evolution(circuit, t):
    circuit.append(RXGate(2 * t), [0])  # Implement e^{-iH_0t}

def perturbation_1(circuit, t):
    circuit.append(RYGate(2 * t), [0])  # Implement e^{-iV_1t}

def perturbation_2(circuit, t):
    circuit.append(RZGate(2 * t), [0])  # Implement e^{-iV_2t}

# Trotterization
for _ in range(n_trotter):
    unperturbed_evolution(qc, t / n_trotter)
    perturbation_1(qc, t / n_trotter)
    perturbation_2(qc, t / n_trotter)

# Draw the circuit
print(qc.draw())
# # Uncomment to get the Circcuit
# a = qc.draw(output='mpl') 
# a.savefig('quantum_circuit.png')  # Save the circuit plot as an image
# print(a)
