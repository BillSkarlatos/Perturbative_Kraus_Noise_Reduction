# heatmaps.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Choi
from qiskit_aer import AerSimulator
from calculations import generate
from kraus import reduce_noise_in_circuit


def plot_heatmap(matrix, title="Heatmap", xlabel="", ylabel="", cmap="viridis"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Choi
from qiskit import transpile
from qiskit_aer.library import save_statevector, save_density_matrix

def simulate_and_plot(qubit_num, shots=1024):
    # Generate the circuit and noise model
    qc, noise_model = generate(qubit_num)

    # Remove measurements explicitly
    qc_no_measure = qc.remove_final_measurements(inplace=False)

    # Ideal backend simulation (statevector)
    qc_ideal = qc_no_measure.copy()
    qc_ideal.save_statevector()

    ideal_backend = AerSimulator(method='statevector')
    ideal_result = ideal_backend.run(transpile(qc_ideal, ideal_backend)).result()
    ideal_state = ideal_result.get_statevector()
    plot_heatmap(np.abs(DensityMatrix(ideal_state).data), "Ideal State Density Matrix")

    # Noisy backend simulation (density matrix)
    qc_noisy = qc_no_measure.copy()
    qc_noisy.save_density_matrix()

    noisy_backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    noisy_result = noisy_backend.run(transpile(qc_noisy, noisy_backend)).result()
    noisy_state = noisy_result.data(0)['density_matrix']
    plot_heatmap(np.abs(noisy_state), "Noisy State Density Matrix")

    # Optimized backend simulation (density matrix)
    optimized_noise_model = reduce_noise_in_circuit(qc_no_measure, noise_model)
    qc_optimized = qc_no_measure.copy()
    qc_optimized.save_density_matrix()

    optimized_backend = AerSimulator(method='density_matrix', noise_model=optimized_noise_model)
    optimized_result = optimized_backend.run(transpile(qc_optimized, optimized_backend)).result()
    optimized_state = optimized_result.data(0)['density_matrix']
    plot_heatmap(np.abs(optimized_state), "Optimized State Density Matrix")

    # Choi matrix visualization
    choi_matrix = Choi(noise_model).data
    plot_heatmap(np.abs(choi_matrix), title="Choi Matrix (Noise Model)", xlabel="Index", ylabel="Index")



if __name__ == '__main__':
    qubit_num = 4  # Example: can be changed.
    simulate_and_plot(qubit_num)