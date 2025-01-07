from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import SuperOp
import numpy as np

# Define the quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Define the noise model
noise_model = NoiseModel()
dep_error_1q = depolarizing_error(0.05, 1)
noise_model.add_all_qubit_quantum_error(dep_error_1q, ['h'])
dep_error_2q = depolarizing_error(0.1, 2)
noise_model.add_all_qubit_quantum_error(dep_error_2q, ['cx'])

# Create the simulator with the noise model
simulator = AerSimulator(noise_model=noise_model)

# Transpile the circuit for the simulator
transpiled_qc = transpile(qc, simulator)

# Generate the SuperOp representation of the noisy circuit
superop = SuperOp(transpiled_qc)
noise_matrix = superop.data

# Compute the pseudo-inverse of the noise matrix
pseudo_inverse = np.linalg.pinv(noise_matrix)

# Verify the pseudo-inverse properties
identity_check = np.allclose(np.dot(noise_matrix, pseudo_inverse), np.eye(noise_matrix.shape[0]))
print(f"Is the pseudo-inverse valid? {identity_check}")
