from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error#, measure_error
import numpy as np
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import QFT

qc = QuantumCircuit(4)
qc.h([0, 1, 2])  # Apply Hadamard gates to qubits
qc.cp(np.pi, 0, 1)  # Controlled phase with pi
qc.cp(np.pi / 2, 1, 2)  # Controlled phase with pi/2
qc.measure_all()

# Define a noise model
noise_model = NoiseModel()

# Add depolarizing noise to all single-qubit gates
depolarizing_1q = depolarizing_error(0.1, 1)  # 10% depolarizing error
noise_model.add_all_qubit_quantum_error(depolarizing_1q, ['h'])

# Add depolarizing noise to two-qubit gates
depolarizing_2q = depolarizing_error(0.03, 2)  # 3% depolarizing error
noise_model.add_all_qubit_quantum_error(depolarizing_2q, ['cp'])


print(qc)
# Latex drawing
try:
    latex_code = circuit_drawer(qc, output="latex_source")
    print("LaTeX source generated successfully.")
    with open("circuit.tex", "w") as f:
        f.write(latex_code)
    print("Exported circuit to circuit.tex. Compile it manually with pdflatex.")
except Exception as e:
    print(f"Error generating LaTeX: {e}")
simulator = AerSimulator()
result = simulator.run(qc, shots=1024).result()
counts = result.get_counts()
print("Measurement outcomes:", counts)

