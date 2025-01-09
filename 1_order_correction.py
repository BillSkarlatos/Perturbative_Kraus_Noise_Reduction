from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import SuperOp, Choi
import numpy as np

def project_to_cptp(choi):
    """
    Project a Choi matrix to the closest CPTP matrix by ensuring eigenvalues are non-negative.
    """
    choi_matrix = choi.data
    # Perform eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(choi_matrix)
    # Set negative eigenvalues to zero
    eigvals[eigvals < 0] = 0
    # Reconstruct the Choi matrix
    projected_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return Choi(projected_matrix)

def apply_noise_correction(qc, noise_model):
    """
    Applies noise correction to a quantum circuit using a perturbative approach.
    """
    # Remove measurement operations for superoperator calculation
    quantum_ops = qc.remove_final_measurements(inplace=False)

    simulator = AerSimulator(noise_model=noise_model)
    transpiled_qc = transpile(quantum_ops, simulator)

    # Generate noise superoperator
    superop = SuperOp(transpiled_qc)
    noise_matrix = superop.data

    # Perturbative correction: Identity - Perturbation
    identity_matrix = np.eye(noise_matrix.shape[0])
    perturbation_matrix = noise_matrix - identity_matrix

    # Apply first-order correction
    correction_matrix = identity_matrix - perturbation_matrix

    # Create a Choi matrix for the correction
    choi = Choi(correction_matrix)
    if not choi.is_cptp():
        print("Correction superoperator is not CPTP. Projecting to CPTP.")
        choi = project_to_cptp(choi)

    # Convert the CPTP Choi matrix back to a SuperOp
    correction_superop = SuperOp(choi)

    # Simulate the noisy circuit
    noisy_qc = quantum_ops.copy()
    noisy_qc.measure_all()
    noisy_result = simulator.run(transpile(noisy_qc, simulator), shots=1024).result()
    noisy_counts = noisy_result.get_counts()

    # Transform noisy probabilities into a diagonal density matrix (vectorized)
    total_shots = sum(noisy_counts.values())
    noisy_probs = {k: v / total_shots for k, v in noisy_counts.items()}
    dim = 2 ** qc.num_qubits
    noisy_density_vector = np.zeros(dim**2)
    for state, prob in noisy_probs.items():
        idx = int(state, 2)
        noisy_density_vector[idx * dim + idx] = prob

    # Apply the correction matrix
    corrected_density_vector = correction_matrix @ noisy_density_vector

    # Extract corrected probabilities
    corrected_probs = {}
    for i in range(dim):
        corrected_probs[f"{i:0{qc.num_qubits}b}"] = corrected_density_vector[i * dim + i]

    # Normalize corrected probabilities
    corrected_probs = {k: max(0, v) for k, v in corrected_probs.items()}  # Clip negatives
    normalization_factor = sum(corrected_probs.values())
    corrected_probs = {k: v / normalization_factor for k, v in corrected_probs.items()}

    # Convert corrected probabilities back to counts
    corrected_counts = {k: int(v * total_shots) for k, v in corrected_probs.items()}

    # Simulate the ideal circuit
    ideal_simulator = AerSimulator()  # No noise model
    ideal_qc = quantum_ops.copy()
    ideal_qc.measure_all()
    ideal_result = ideal_simulator.run(transpile(ideal_qc, ideal_simulator), shots=1024).result()
    ideal_counts = ideal_result.get_counts()


    # Calculate differences
    def calculate_difference(counts1, counts2):
        total_shots = sum(counts1.values())
        probs1 = {k: v / total_shots for k, v in counts1.items()}
        probs2 = {k: v / total_shots for k, v in counts2.items()}
        return {key: abs(probs1.get(key, 0) - probs2.get(key, 0)) for key in probs1.keys() | probs2.keys()}

    differences = calculate_difference(ideal_counts, corrected_counts)

    return noisy_counts, corrected_counts, ideal_counts, differences

# Define the complex 4-qubit circuit
qc = QuantumCircuit(4, 4)

# Apply Hadamard gates to all qubits to create superposition
for qubit in range(4):
    qc.h(qubit)

# Add controlled-X (CNOT) gates to entangle qubits
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# Apply parameterized rotations for more complexity
theta = np.pi / 3
qc.rx(theta, 0)
qc.ry(theta / 2, 1)
qc.rz(theta / 4, 2)

# Add a controlled-controlled-Z (CCZ) gate
qc.h(3)
qc.ccx(0, 1, 3)  # Control qubits 0 and 1, target qubit 3
qc.h(3)

# Add additional single-qubit gates
qc.x(2)
qc.y(3)

# Measure all qubits
qc.measure(range(4), range(4))

# Display the circuit
print("Circuit:")
print(qc)

# Define the noise parameters
p1 = 0.1  # 1-qubit gate error
p2 = 2 * p1  # 2-qubit gate error

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
therm_error_3q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
    thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )
)
dep_error_3q = depolarizing_error(p2, 3)
error_3q = dep_error_3q.compose(therm_error_3q)

# Combine errors
error_1q = dep_error_1q.compose(therm_error_1q)
error_2q = dep_error_2q.compose(therm_error_2q)

# Build the noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz', 'x', 'y'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
noise_model.add_all_qubit_quantum_error(error_3q, ['ccx'])

# Apply noise correction
noisy_counts, corrected_counts, ideal_counts, differences = apply_noise_correction(qc, noise_model)

# Print results in a more understandable way
print("Results:")
print("Noisy Counts:")
for state, count in sorted(noisy_counts.items()):
    print(f"  State {state}: {count}")

print("\nCorrected Counts:")
for state, count in sorted(corrected_counts.items()):
    print(f"  State {state}: {count}")

print("\nIdeal Counts:")
for state, count in sorted(ideal_counts.items()):
    print(f"  State {state}: {count}")

print("\nDifferences (Probability):")
for state, diff in sorted(differences.items()):
    print(f"  State {state}: {diff:.4f}")
