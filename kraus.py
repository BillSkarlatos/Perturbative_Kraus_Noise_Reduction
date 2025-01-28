from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from qiskit.quantum_info import Choi, Kraus
import numpy as np
import cvxpy as cp
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from calculations import *

# Function to plot results as scatter plots
def plot_results_as_dots(results, labels, title, label_step=10):
    """
    Plot quantum simulation results as scatter plots, sorted by bitstrings, with properly aligned x-axis labels.

    Args:
    - results: List of dictionaries containing simulation results.
    - labels: List of strings for the labels of each dataset.
    - title: Title of the plot.
    - label_step: Interval for showing x-axis labels (default is 10).
    """
    plt.figure(figsize=(12, 8))
    marker_styles = ['o', 's', '^']  # Different markers for each dataset

    # Process the results to ensure sorting by bitstrings
    sorted_results = []
    all_bitstrings = set()  # To gather all possible bitstrings for consistent alignment
    for result in results:
        sorted_items = sorted(result.items())  # Sort by bitstring keys
        sorted_results.append(sorted_items)
        all_bitstrings.update(result.keys())

    # Sort all bitstrings globally for alignment
    all_bitstrings = sorted(all_bitstrings)

    # Plot data for each result
    for i, sorted_result in enumerate(sorted_results):
        # Create full y-data aligned with all_bitstrings
        y_data = [dict(sorted_result).get(bitstring, 0) for bitstring in all_bitstrings]
        plt.scatter(range(len(all_bitstrings)), y_data, label=labels[i],
                    marker=marker_styles[i % len(marker_styles)], s=100)

    # Customize x-axis labels
    plt.xticks(
        ticks=range(0, len(all_bitstrings), label_step),  # Label every `label_step` ticks
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

qubit_num=2

qc,noise_model=generate(qubit_num)
print(qc.draw())

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
# print("Raw results from simulation:")
# print("Noisy counts:", counts_noisy)
# print("Optimized counts:", counts_optimized)
# print("Ideal counts:", counts_ideal)

shots=1024

# Transform results into dictionaries for visualization
counts_noisy_dict = {str(k): v for k, v in counts_noisy.items()}
counts_optimized_dict = {str(k): v for k, v in counts_optimized.items()}
counts_ideal_dict = {str(k): v for k, v in counts_ideal.items()}

normalized_counts_noisy = normalize_counts(counts_noisy_dict, shots)
normalized_counts_optimized = normalize_counts(counts_optimized_dict, shots)
normalized_counts_ideal = normalize_counts(counts_ideal_dict, shots)

# Ensure alignment between printed and plotted data
assert counts_noisy_dict == {str(k): v for k, v in counts_noisy.items()}, "Noisy data mismatch!"
assert counts_optimized_dict == {str(k): v for k, v in counts_optimized.items()}, "Optimized data mismatch!"
assert counts_ideal_dict == {str(k): v for k, v in counts_ideal.items()}, "Ideal data mismatch!"

print("Fidelity of noisy system: ", calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_noisy))
print("Fidelity of corrected system: ", calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_optimized))
print("Ratio (ideal/noisy): ",calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_optimized)/calculate_fidelity(ideal_counts=normalized_counts_ideal, test_counts=normalized_counts_noisy))

# Prepare data for plotting
results = [normalized_counts_noisy, normalized_counts_optimized, normalized_counts_ideal]
labels = ["Noisy Model", "Optimized Noise Model", "Ideal Model"]

# Plot the results as scatter plots
step=1
if (qubit_num >= 16):
    step = int(6*qubit_num)
elif (qubit_num >=12):
    step = int(3*qubit_num)
elif (qubit_num >= 8):
    step = int(1.5*qubit_num)
elif (qubit_num >= 4):
    step = int(0.75*qubit_num)
plot_results_as_dots(results, labels, "Comparison of Simulation Results", step)




