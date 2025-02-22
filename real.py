from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
from qiskit.quantum_info import Choi, Kraus
import numpy as np
import cvxpy as cp
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from calculations import *

# Initialize IBM Quantum service
service = QiskitRuntimeService()
backend = service.backend('ibm_brisbane')  # Replace with an available backend
sampler = Sampler(backend)

# Extract real noise model from backend
noise_model = NoiseModel.from_backend(backend)

def reduce_noise_in_circuit(circuit, noise_model):
    """Reduce noise in a given quantum circuit."""
    optimized_noise_model = NoiseModel()
    for instr in noise_model.noise_instructions:
        for qubits in noise_model.qubits_with_instruction(instr):
            quantum_error = noise_model.instruction_errors(instr, qubits)[0]
            kraus_ops = quantum_error.to_kraus()
            choi_matrix = Choi(Kraus(kraus_ops)).data
            if not is_cptp(choi_matrix):
                choi_matrix = cptp_projection(choi_matrix)
            optimized_kraus = optimize_kraus_operators(kraus_ops)
            optimized_quantum_error = Kraus(optimized_kraus)
            optimized_noise_model.add_quantum_error(optimized_quantum_error, instr, qubits)
    return optimized_noise_model

# Function to plot results as scatter plots
def plot_results_as_dots(results, labels, title, label_step=10):
    plt.figure(figsize=(12, 8))
    marker_styles = ['o', 's', '^']  # Different markers for each dataset
    sorted_results = []
    all_bitstrings = set()
    for result in results:
        sorted_items = sorted(result.items())
        sorted_results.append(sorted_items)
        all_bitstrings.update(result.keys())
    all_bitstrings = sorted(all_bitstrings)
    for i, sorted_result in enumerate(sorted_results):
        y_data = [dict(sorted_result).get(bitstring, 0) for bitstring in all_bitstrings]
        plt.scatter(range(len(all_bitstrings)), y_data, label=labels[i],
                    marker=marker_styles[i % len(marker_styles)], s=100)
    plt.xticks(
        ticks=range(0, len(all_bitstrings), label_step),
        labels=[all_bitstrings[i] for i in range(0, len(all_bitstrings), label_step)],
        rotation=45,
        ha='right'
    )
    plt.xlabel("Bitstrings")
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Generate quantum circuit
qubit_num = 4
qc = QuantumCircuit(qubit_num)
qc.h(range(qubit_num))
qc.measure_all()

# Transpile circuit for real hardware
transpiled_circuit = transpile(qc, backend, optimization_level=3)

# Execute on real hardware using Sampler primitive
job_real = sampler.run([transpiled_circuit], shots=1024)
result_real = job_real.result()
counts_real = result_real.quasi_dists[0]

# Optimize the noise model
optimized_noise_model = reduce_noise_in_circuit(qc, noise_model)

# Execute optimized circuit on real hardware
transpiled_circuit_opt = transpile(qc, backend, optimization_level=3)
job_opt = sampler.run([transpiled_circuit_opt], shots=1024)
result_opt = job_opt.result()
counts_optimized = result_opt.quasi_dists[0]

# Simulate ideal (no-noise) execution
ideal_backend = AerSimulator()
transpiled_circuit_ideal = transpile(qc, ideal_backend)
job_ideal = ideal_backend.run(transpiled_circuit_ideal, shots=1024)
result_ideal = job_ideal.result()
counts_ideal = result_ideal.get_counts()

# Normalize results
shots = 1024
counts_real_dict = {str(k): v for k, v in counts_real.items()}
counts_optimized_dict = {str(k): v for k, v in counts_optimized.items()}
counts_ideal_dict = {str(k): v for k, v in counts_ideal.items()}
normalized_counts_real = normalize_counts(counts_real_dict, shots)
normalized_counts_optimized = normalize_counts(counts_optimized_dict, shots)
normalized_counts_ideal = normalize_counts(counts_ideal_dict, shots)

# Fidelity comparison
print("Fidelity of real hardware (before optimization):", calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_real))
print("Fidelity after Kraus optimization:", calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_optimized))
print("Ratio (ideal/noisy):", calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_optimized) / calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_real))

# Plot results
results = [normalized_counts_real, normalized_counts_optimized, normalized_counts_ideal]
labels = ["Real Hardware", "Optimized Noise Model", "Ideal Model"]
plot_results_as_dots(results, labels, "Comparison of Real Hardware Execution")
