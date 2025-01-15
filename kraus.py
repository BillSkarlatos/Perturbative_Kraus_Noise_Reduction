from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import Choi, Kraus
import numpy as np
import cvxpy as cp
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

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

# Step 1: Create a quantum circuit
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Step 2: Define a noise model
noise_model = NoiseModel()

# Add depolarizing error
depol_error = depolarizing_error(0.01, 1)
instructions = ['id', 'u1', 'u2', 'u3']
for instr in instructions:
    if instr not in noise_model.noise_instructions:
        noise_model.add_all_qubit_quantum_error(depol_error, instr)

# Add thermal relaxation error
thermal_error = thermal_relaxation_error(50e-6, 30e-6, 0.01)
for instr in instructions:
    if instr not in noise_model.noise_instructions:
        noise_model.add_all_qubit_quantum_error(thermal_error, instr)

# Step 3: Simulate the circuit with the noisy model
noisy_backend = AerSimulator(noise_model=noise_model)
transpiled_circuit_noisy = transpile(qc, noisy_backend)
job_noisy = noisy_backend.run(transpiled_circuit_noisy, shots=1024)
result_noisy = job_noisy.result()
counts_noisy = result_noisy.get_counts()

# Step 4: Optimize the noise model
optimized_noise_model = reduce_noise_in_circuit(qc, noise_model)

# Step 5: Simulate the circuit with the optimized noise model
optimized_backend = AerSimulator(noise_model=optimized_noise_model)
transpiled_circuit_optimized = transpile(qc, optimized_backend)
job_optimized = optimized_backend.run(transpiled_circuit_optimized, shots=1024)
result_optimized = job_optimized.result()
counts_optimized = result_optimized.get_counts()

# Step 6: Simulate the ideal circuit
ideal_backend = AerSimulator()
transpiled_circuit_ideal = transpile(qc, ideal_backend)
job_ideal = ideal_backend.run(transpiled_circuit_ideal, shots=1024)
result_ideal = job_ideal.result()
counts_ideal = result_ideal.get_counts()

# Step 7: Output results
print("Simulation results with noisy model:")
print(counts_noisy)
print("\nSimulation results with optimized noise model:")
print(counts_optimized)
print("\nSimulation results with ideal model:")
print(counts_ideal)

# Step 8: Compare results visually
counts_noisy_dict = {str(k): v for k, v in counts_noisy.items()}
counts_optimized_dict = {str(k): v for k, v in counts_optimized.items()}
counts_ideal_dict = {str(k): v for k, v in counts_ideal.items()}

# Flatten the results into a single dictionary for compatibility
results_flattened = {
    "Noisy Model": counts_noisy_dict,
    "Optimized Noise Model": counts_optimized_dict,
    "Ideal Model": counts_ideal_dict
}

# Plot comparison
plot_histogram([counts_noisy_dict, counts_optimized_dict, counts_ideal_dict],
               legend=["Noisy Model", "Optimized Noise Model", "Ideal Model"],
               title="Comparison of Simulation Results", bar_labels=True)
plt.show()

