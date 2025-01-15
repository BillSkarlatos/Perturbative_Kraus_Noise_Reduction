from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import SuperOp
import numpy as np
import matplotlib.pyplot as plt

def apply_noise_correction(qc, noise_model):
    """
    Applies noise correction to a quantum circuit using a perturbative approach.
    """
    # Remove measurement operations for SuperOp calculation
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

    # Simulate noisy circuit
    noisy_qc = quantum_ops.copy()
    noisy_qc.measure_all()
    noisy_result = simulator.run(transpile(noisy_qc, simulator), shots=1024).result()
    noisy_counts = noisy_result.get_counts()

    # Transform noisy probabilities into a diagonal density matrix (vectorized)
    total_shots = sum(noisy_counts.values())
    noisy_probs = {k: v / total_shots for k, v in noisy_counts.items()}
    dim = 2 ** quantum_ops.num_qubits
    noisy_density_vector = np.zeros(dim**2)
    for state, prob in noisy_probs.items():
        idx = int(state.replace(" ", ""), 2)
        noisy_density_vector[idx * dim + idx] = prob

    # Apply correction
    corrected_density_vector = correction_matrix @ noisy_density_vector
    corrected_probs = {f"{i:0{quantum_ops.num_qubits}b}": corrected_density_vector[i * dim + i]
                       for i in range(dim)}
    normalization_factor = sum(corrected_probs.values())
    corrected_probs = {k: max(0, v) / normalization_factor for k, v in corrected_probs.items()}
    corrected_counts = {k: int(v * total_shots) for k, v in corrected_probs.items()}

    # Simulate ideal circuit
    ideal_simulator = AerSimulator()
    ideal_qc = quantum_ops.copy()
    ideal_qc.measure_all()
    ideal_result = ideal_simulator.run(transpile(ideal_qc, ideal_simulator), shots=1024).result()
    ideal_counts = ideal_result.get_counts()

    return noisy_counts, corrected_counts, ideal_counts

# Quantum Internet Circuit (Teleportation)
qc = QuantumCircuit(3, 2)
qc.h(0)
qc.cx(0, 1)
qc.h(2)
qc.cx(2, 0)
qc.h(2)
qc.measure([0, 2], [0, 1])

# Noise Model Setup
p1 = 0.1
p2 = 2 * p1
t1 = 50e3
t2 = 70e3
gate_time_1q = 50
gate_time_2q = 150

dep_error_1q = depolarizing_error(p1, 1)
dep_error_2q = depolarizing_error(p2, 2)
therm_error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
therm_error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
    thermal_relaxation_error(t1, t2, gate_time_2q)
)

error_1q = dep_error_1q.compose(therm_error_1q)
error_2q = dep_error_2q.compose(therm_error_2q)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1q, ['h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Apply noise correction
noisy_counts, corrected_counts, ideal_counts = apply_noise_correction(qc, noise_model)

# Results
print("Noisy Counts:", noisy_counts)
print("Corrected Counts:", corrected_counts)
print("Ideal Counts:", ideal_counts)

# Calculate percentage difference
def calculate_percentage_difference(measured_counts, ideal_counts):
    total_ideal = sum(ideal_counts.values())
    percentage_differences = {}
    for state in ideal_counts:
        ideal_value = ideal_counts.get(state, 0) / total_ideal
        measured_value = measured_counts.get(state, 0) / total_ideal
        percentage_differences[state] = abs(measured_value - ideal_value) / ideal_value * 100 if ideal_value > 0 else 0
    return percentage_differences

def calculate_average_percentage_difference(percentage_differences):
    if not percentage_differences:
        return 0
    return sum(percentage_differences.values()) / len(percentage_differences)

noisy_diff = calculate_percentage_difference(noisy_counts, ideal_counts)
corrected_diff = calculate_percentage_difference(corrected_counts, ideal_counts)

average_noisy_diff = calculate_average_percentage_difference(noisy_diff)
average_corrected_diff = calculate_average_percentage_difference(corrected_diff)

print("\nPercentage Difference (Noisy vs Ideal):")
for state, diff in noisy_diff.items():
    print(f"  State {state}: {diff:.2f}%")

print(f"Average Percentage Difference (Noisy vs Ideal): {average_noisy_diff:.2f}%")

print("\nPercentage Difference (Corrected vs Ideal):")
for state, diff in corrected_diff.items():
    print(f"  State {state}: {diff:.2f}%")

print(f"Average Percentage Difference (Corrected vs Ideal): {average_corrected_diff:.2f}%")

# Visualize results with matplotlib
states = sorted(set(noisy_counts.keys()).union(corrected_counts.keys(), ideal_counts.keys()))
noisy_values = [noisy_counts.get(state, 0) for state in states]
corrected_values = [corrected_counts.get(state, 0) for state in states]
ideal_values = [ideal_counts.get(state, 0) for state in states]

x = range(len(states))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width for i in x], noisy_values, width, label="Noisy Counts", alpha=0.75)
ax.bar(x, corrected_values, width, label="Corrected Counts", alpha=0.75)
ax.bar([i + width for i in x], ideal_values, width, label="Ideal Counts", alpha=0.75)

ax.set_title("Counts Comparison")
ax.set_xlabel("State")
ax.set_ylabel("Counts")
ax.set_xticks(x)
ax.set_xticklabels(states, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()