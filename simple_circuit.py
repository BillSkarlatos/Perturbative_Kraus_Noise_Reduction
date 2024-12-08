from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, HGate, CXGate, SwapGate
import numpy as np
import matplotlib.pyplot as plt

# Parameters
t = 1.0  # Evolution time
n_trotter = 5  # Number of Trotter steps

# Define the circuit with 3 qubits (more complex than just 1 qubit)
qc = QuantumCircuit(3)

# Define time evolution for H0, V1, and V2
def unperturbed_evolution(circuit, t):
    circuit.append(RXGate(2 * t), [0])  # Implement e^{-iH_0t} on qubit 0
    circuit.append(RYGate(2 * t), [1])  # Implement e^{-iH_0t} on qubit 1

def perturbation_1(circuit, t):
    circuit.append(RZGate(2 * t), [1])  # Implement e^{-iV_1t} on qubit 1
    circuit.append(HGate(), [2])        # Apply Hadamard on qubit 2

def perturbation_2(circuit, t):
    circuit.append(CXGate(), [0, 1])  # Entangle qubits 0 and 1 using CNOT
    circuit.append(SwapGate(), [1, 2])  # Swap qubits 1 and 2

# Trotterization
for _ in range(n_trotter):
    unperturbed_evolution(qc, t / n_trotter)
    perturbation_1(qc, t / n_trotter)
    perturbation_2(qc, t / n_trotter)

# Add a measurement step (optional for testing purposes)
qc.measure_all()

# Draw the circuit (text-based representation)
print(qc.draw())

# Uncomment to save the circuit plot as an image
a = qc.draw(output='mpl')  # Generate the matplotlib figure object
a.savefig('QC.png')  # Save the circuit plot as an image
plt.show()  # Display the plot
