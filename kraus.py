from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from qiskit.quantum_info import Choi, Kraus
import numpy as np
import cvxpy as cp
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Function to plot results as scatter plots
def plot_results_as_dots(results, labels, title):
    """
    Plot quantum simulation results as scatter plots.

    Args:
    - results: List of dictionaries containing simulation results.
    - labels: List of strings for the labels of each dataset.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    marker_styles = ['o', 's', '^']  # Different markers for each dataset

    # Iterate through each result set
    for i, result in enumerate(results):
        x = list(result.keys())  # Extract bitstrings
        y = list(result.values())  # Extract counts
        plt.scatter(x, y, label=labels[i], marker=marker_styles[i % len(marker_styles)], s=100)

    plt.xlabel("Bitstrings")
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def is_cptp(choi_matrix):
    """Check if a Choi matrix represents a CPTP map."""
    eigenvalues = np.linalg.eigvals(choi_matrix)
    trace = np.trace(choi_matrix)
    return np.all(eigenvalues >= 0) and np.isclose(trace, choi_matrix.shape[0])

def cptp_projection(choi_matrix):
    """Project a Choi matrix to the nearest CPTP map."""
    dim = choi_matrix.shape[0]
    choi_cp = cp.Variable((dim, dim), complex=True)
    constraints = [
        choi_cp >> 0,  # Positive semidefinite
        cp.trace(choi_cp) == dim  # Trace preserving
    ]
    problem = cp.Problem(cp.Minimize(cp.norm(choi_cp - choi_matrix, 'fro')), constraints)
    problem.solve()
    return choi_cp.value

def optimize_kraus_operators(kraus_ops):
    """Optimize Kraus operators to reduce noise."""
    optimized_kraus = []
    for op in kraus_ops:
        optimized_op = op / np.linalg.norm(op)  # Normalize Kraus operators
        optimized_kraus.append(optimized_op)
    return optimized_kraus

def reduce_noise_in_circuit(circuit, noise_model):
    """Reduce noise in a given quantum circuit."""
    optimized_noise_model = NoiseModel()

    for instr in noise_model.noise_instructions:
        if instr in noise_model._local_quantum_errors:  # Check if the instruction has local errors
            for qubits, quantum_error in noise_model._local_quantum_errors[instr].items():
                kraus_ops = quantum_error.to_kraus()
                choi_matrix = Choi(quantum_error).data

                # Ensure CPTP
                if not is_cptp(choi_matrix):
                    choi_matrix = cptp_projection(choi_matrix)

                # Optimize Kraus operators
                optimized_kraus = optimize_kraus_operators(kraus_ops)

                # Create a new QuantumError with optimized Kraus operators
                optimized_quantum_error = Kraus(optimized_kraus)

                # Add the optimized error to the new noise model
                optimized_noise_model.add_quantum_error(optimized_quantum_error, instr, qubits)

    return optimized_noise_model

from qiskit import QuantumCircuit

# Create a 12-qubit quantum circuit
qc = QuantumCircuit(12)

# Apply Hadamard gates to create superposition on the first 3 qubits
qc.h(0)
qc.h(1)
qc.h(2)

# Apply a series of controlled NOT gates (entangling qubits)
qc.cx(0, 3)  # CNOT gate on qubit 0 and qubit 3
qc.cx(1, 4)  # CNOT gate on qubit 1 and qubit 4
qc.cx(2, 5)  # CNOT gate on qubit 2 and qubit 5
qc.cx(3, 6)  # CNOT gate on qubit 3 and qubit 6
qc.cx(4, 7)  # CNOT gate on qubit 4 and qubit 7
qc.cx(5, 8)  # CNOT gate on qubit 5 and qubit 8
qc.cx(6, 9)  # CNOT gate on qubit 6 and qubit 9
qc.cx(7, 10) # CNOT gate on qubit 7 and qubit 10
qc.cx(8, 11) # CNOT gate on qubit 8 and qubit 11

# Apply a Hadamard gate to qubit 11 to introduce interference
qc.h(11)

# Apply a controlled gate for some additional complexity
qc.ccx(0, 1, 2)  # Toffoli gate (CCX gate) on qubits 0, 1, 2

# Measure the qubits
qc.measure_all()

# Visualize the circuit
qc.draw('mpl')


# Step 2: Define the noise model
noise_model = NoiseModel()

# Add depolarizing error
depol_error_1q = depolarizing_error(0.01, 1)  # 1% depolarizing noise for 1-qubit gates
depol_error_2q = depolarizing_error(0.025, 2)  # 2.5% depolarizing noise for 2-qubit gates
depol_error_3q = depolarizing_error(0.035, 3)  # 3.5% depolarizing noise for 2-qubit gates

# Add thermal relaxation error
thermal_error_1q = thermal_relaxation_error(t1=50e-6, t2=30e-6, time=20e-6)
thermal_error_2q = thermal_relaxation_error(t1=50e-6, t2=30e-6, time=40e-6)
thermal_error_3q = thermal_relaxation_error(t1=50e-6, t2=30e-6, time=80e-6)

# Add amplitude damping error
amp_damp_error = amplitude_damping_error(0.05)  # 5% probability of amplitude damping

# Combine errors (composite errors)
composite_1q_error = depol_error_1q.compose(thermal_error_1q)
composite_2q_error = depol_error_2q.compose(thermal_error_2q)
composite_3q_error = depol_error_3q.compose(thermal_error_3q)

# Step 3: Add errors to specific gates in the noise model
# Single-qubit gates
noise_model.add_all_qubit_quantum_error(composite_1q_error, ['u1', 'u2', 'u3', 'id'])
# Two-qubit gates
noise_model.add_all_qubit_quantum_error(composite_2q_error, ['cx'])
# Three-qubit gates
noise_model.add_all_qubit_quantum_error(composite_3q_error, ['ccx'])
# Amplitude damping specifically to `id` gate
noise_model.add_all_qubit_quantum_error(amp_damp_error, ['id'])

# Step 4: Simulate the circuit with the noisy model
noisy_backend = AerSimulator(noise_model=noise_model)
transpiled_circuit_noisy = transpile(qc, noisy_backend)
job_noisy = noisy_backend.run(transpiled_circuit_noisy, shots=1024)
result_noisy = job_noisy.result()
counts_noisy = result_noisy.get_counts()

# Step 5: Optimize the noise model
optimized_noise_model = reduce_noise_in_circuit(qc, noise_model)

# Step 6: Simulate the circuit with the optimized noise model
optimized_backend = AerSimulator(noise_model=optimized_noise_model)
transpiled_circuit_optimized = transpile(qc, optimized_backend)
job_optimized = optimized_backend.run(transpiled_circuit_optimized, shots=1024)
result_optimized = job_optimized.result()
counts_optimized = result_optimized.get_counts()

# Step 7: Simulate the ideal circuit
ideal_backend = AerSimulator()
transpiled_circuit_ideal = transpile(qc, ideal_backend)
job_ideal = ideal_backend.run(transpiled_circuit_ideal, shots=1024)
result_ideal = job_ideal.result()
counts_ideal = result_ideal.get_counts()

# Debugging step: Ensure consistency in printed and plotted data
print("Raw results from simulation:")
print("Noisy counts:", counts_noisy)
print("Optimized counts:", counts_optimized)
print("Ideal counts:", counts_ideal)

# Transform results into dictionaries for visualization
counts_noisy_dict = {str(k): v for k, v in counts_noisy.items()}
counts_optimized_dict = {str(k): v for k, v in counts_optimized.items()}
counts_ideal_dict = {str(k): v for k, v in counts_ideal.items()}

# # Debugging step: Verify transformed data
# print("\nTransformed data for plotting:")
# print("Noisy counts (dict):", counts_noisy_dict)
# print("Optimized counts (dict):", counts_optimized_dict)
# print("Ideal counts (dict):", counts_ideal_dict)

# Ensure alignment between printed and plotted data
assert counts_noisy_dict == {str(k): v for k, v in counts_noisy.items()}, "Noisy data mismatch!"
assert counts_optimized_dict == {str(k): v for k, v in counts_optimized.items()}, "Optimized data mismatch!"
assert counts_ideal_dict == {str(k): v for k, v in counts_ideal.items()}, "Ideal data mismatch!"


# Prepare data for plotting
results = [counts_noisy_dict, counts_optimized_dict, counts_ideal_dict]
labels = ["Noisy Model", "Optimized Noise Model", "Ideal Model"]

# Plot the results as scatter plots
plot_results_as_dots(results, labels, "Comparison of Simulation Results")




