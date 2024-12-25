import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import mthree

# Step 1: Create the Quantum Circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Step 2: Define a Noise Model
# Depolarizing error probabilities
p1 = 0.05  # Increased 1-qubit gate error
p2 = 0.1   # Increased 2-qubit gate error

# Thermal relaxation parameters
t1 = 50e3  # T1 time in nanoseconds
t2 = 70e3  # T2 time in nanoseconds
gate_time_1q = 50  # 1-qubit gate time in nanoseconds
gate_time_2q = 150  # 2-qubit gate time in nanoseconds

# Create depolarizing errors
dep_error_1q = depolarizing_error(p1, 1)
dep_error_2q = depolarizing_error(p2, 2)

# Create thermal relaxation errors
therm_error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
therm_error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
    thermal_relaxation_error(t1, t2, gate_time_2q)
)

# Combine errors
error_1q = dep_error_1q.compose(therm_error_1q)
error_2q = dep_error_2q.compose(therm_error_2q)

# Build the noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Define and add measurement error
readout_error = ReadoutError([[0.9, 0.1], [0.2, 0.8]])
noise_model.add_all_qubit_readout_error(readout_error)

# Step 3: Simulate the Circuit with Noise
simulator = AerSimulator(noise_model=noise_model)
transpiled_qc = transpile(qc, simulator)
noisy_result = simulator.run(transpiled_qc, shots=1024).result()
noisy_counts = noisy_result.get_counts()

# Step 4: Simulate the Ideal Circuit
ideal_simulator = AerSimulator()
ideal_result = ideal_simulator.run(transpiled_qc, shots=1024).result()
ideal_counts = ideal_result.get_counts()

# Step 5: Apply Measurement Error Mitigation using Mthree
# Initialize Mthree mitigator
mit = mthree.M3Mitigation(simulator)

# Provide the qubits explicitly (list of qubit indices)
print("Calibrating mitigator...")
mit.cals_from_system(qubits=[0, 1])

# Debug: Generate calibration matrix
calibration_matrix = mit.cals_to_matrices()
print("Calibration matrix:")
print(calibration_matrix)

# Debug: Print noisy counts
print("Noisy counts before mitigation:")
print(noisy_counts)

# Apply mitigation to the noisy counts
mitigated_counts = mit.apply_correction(noisy_counts, qubits=[0, 1])

# Debug: Print mitigated counts
print("Mitigated counts after correction:")
print(mitigated_counts)

# Convert mitigated probabilities to raw counts
scaled_counts = {k: int(v * 1024) for k, v in mitigated_counts.items()}
print("Scaled mitigated counts:", scaled_counts)

# Step 6: Compare Results
plt.figure(figsize=(12, 4))

# Plot ideal counts
plt.subplot(1, 3, 1)
plot_histogram(ideal_counts, title="Ideal Counts", ax=plt.gca())

# Plot noisy counts
plt.subplot(1, 3, 2)
plot_histogram(noisy_counts, title="Noisy Counts", ax=plt.gca())

# Plot mitigated counts with adjusted scaling
plt.subplot(1, 3, 3)
plot_histogram(scaled_counts, title="Mitigated Counts", ax=plt.gca(), bar_labels=True)

plt.tight_layout()
plt.show()