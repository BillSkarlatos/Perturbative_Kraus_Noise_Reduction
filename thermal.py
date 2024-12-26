import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import mthree

def analyze_noise_vs_ideal_fixed(n_qubits, shots=8192, calibration_shots=65536):
    """
    Analyzes the loss percentage and fidelity of noisy and mitigated circuits compared to the ideal circuit
    for a range of qubit numbers from 1 to n_qubits. Includes fixes for scaling and normalization issues.
    """
    qubit_range = range(1, n_qubits + 1)
    losses_noisy = []
    losses_mitigated = []
    fidelities_noisy = []
    fidelities_mitigated = []

    for n in qubit_range:
        # Step 1: Create the Quantum Circuit
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.h(i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))

        # Step 2: Define a Noise Model
        p1 = 0.005
        p2 = 0.01
        t1 = 150e3
        t2 = 100e3
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
        readout_error = ReadoutError([[0.99, 0.01], [0.01, 0.99]])
        noise_model.add_all_qubit_readout_error(readout_error)

        # Step 3: Simulate the Circuit with Noise
        simulator = AerSimulator(noise_model=noise_model)
        transpiled_qc = transpile(qc, simulator, optimization_level=3)
        noisy_result = simulator.run(transpiled_qc, shots=shots).result()
        noisy_counts = noisy_result.get_counts()

        # Step 4: Simulate the Ideal Circuit
        ideal_simulator = AerSimulator()
        ideal_result = ideal_simulator.run(transpiled_qc, shots=shots).result()
        ideal_counts = ideal_result.get_counts()

        # Calculate noisy loss and fidelity
        ideal_total = sum(ideal_counts.values())
        noisy_loss = sum(abs(ideal_counts.get(k, 0) - noisy_counts.get(k, 0)) for k in ideal_counts) / ideal_total
        losses_noisy.append(noisy_loss * 100)
        noisy_fidelity = (sum(np.sqrt(ideal_counts.get(k, 0) * noisy_counts.get(k, 0)) for k in ideal_counts) / ideal_total) ** 2
        fidelities_noisy.append(noisy_fidelity)

        # Step 5: Apply Measurement Error Mitigation
        mit = mthree.M3Mitigation(simulator)
        mit.cals_from_system(qubits=list(range(n)), shots=calibration_shots)
        mitigated_counts = mit.apply_correction(noisy_counts, qubits=list(range(n)))

        # Normalize mitigated counts
        mitigated_total = sum(mitigated_counts.values())
        if mitigated_total > 0:
            mitigated_counts = {k: v / mitigated_total for k, v in mitigated_counts.items()}

        # Calculate mitigated loss and fidelity
        mitigated_loss = sum(abs(ideal_counts.get(k, 0) - (mitigated_counts.get(k, 0) * ideal_total)) for k in ideal_counts) / ideal_total
        losses_mitigated.append(mitigated_loss * 100)
        mitigated_fidelity = (sum(np.sqrt(ideal_counts.get(k, 0) * mitigated_counts.get(k, 0)) for k in ideal_counts) / ideal_total) ** 2
        fidelities_mitigated.append(mitigated_fidelity)

    # Calculate average metrics
    avg_loss_noisy = np.mean(losses_noisy)
    avg_loss_mitigated = np.mean(losses_mitigated)
    avg_fidelity_noisy = np.mean(fidelities_noisy)
    avg_fidelity_mitigated = np.mean(fidelities_mitigated)

    print(f"Average Noisy Loss: {avg_loss_noisy:.2f}%")
    print(f"Average Mitigated Loss: {avg_loss_mitigated:.2f}%")
    print(f"Average Noisy Fidelity: {avg_fidelity_noisy:.2f}")
    print(f"Average Mitigated Fidelity: {avg_fidelity_mitigated:.2f}")

    # Step 6: Plot Results
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_range, losses_noisy, label="Noisy Loss (%)", marker='o')
    plt.plot(qubit_range, losses_mitigated, label="Mitigated Loss (%)", marker='o')
    plt.xlabel("Number of Qubits")
    plt.ylabel("Loss Percentage (%)")
    plt.title("Loss Percentage: Noisy vs Mitigated Circuits")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(qubit_range, fidelities_noisy, label="Noisy Fidelity", marker='o')
    plt.plot(qubit_range, fidelities_mitigated, label="Mitigated Fidelity", marker='o')
    plt.xlabel("Number of Qubits")
    plt.ylabel("Fidelity")
    plt.title("Fidelity: Noisy vs Mitigated Circuits")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
analyze_noise_vs_ideal_fixed(10)

