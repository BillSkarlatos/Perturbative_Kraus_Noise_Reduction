from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# Define the quantum circuit
qc = QuantumCircuit(3)
qc.rx(0.4, 0)
qc.ry(0.4, 1)
qc.h(2)
qc.cx(0, 1)
qc.cx(1, 2)
qc.rx(0.4, 0)
qc.ry(0.4, 1)
qc.rz(0.4, 2)
qc.cx(0, 1)
qc.cx(1, 2)
qc.rx(0.4, 0)
qc.ry(0.4, 1)
qc.rz(0.4, 2)
qc.measure_all()

# Create noise model
noise_model = NoiseModel()
error_1q = depolarizing_error(0.01, 1)  # 1% depolarizing noise
error_2q = depolarizing_error(0.02, 2)  # 2% depolarizing noise
noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Define simulators
ideal_simulator = AerSimulator()  # No noise
noisy_simulator = AerSimulator(noise_model=noise_model)  # With noise

# Transpile the circuit for the simulators
ideal_circuit = transpile(qc, ideal_simulator)
noisy_circuit = transpile(qc, noisy_simulator)

# Run ideal simulation
ideal_job = ideal_simulator.run(ideal_circuit, shots=1024)
ideal_result = ideal_job.result()
ideal_counts = ideal_result.get_counts()

# Run noisy simulation without perturbation
noisy_job = noisy_simulator.run(noisy_circuit, shots=1024)
noisy_result = noisy_job.result()
noisy_counts = noisy_result.get_counts()

# Run noisy simulation with perturbation
# For demonstration, mimic perturbation effect by adjusting noise (simplified for illustration)
perturbed_noise_model = NoiseModel()
perturbed_noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ['rx', 'ry', 'rz', 'h'])
perturbed_noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])
perturbed_simulator = AerSimulator(noise_model=perturbed_noise_model)
perturbed_circuit = transpile(qc, perturbed_simulator)
perturbed_job = perturbed_simulator.run(perturbed_circuit, shots=1024)
perturbed_result = perturbed_job.result()
perturbed_counts = perturbed_result.get_counts()

# Plot all results
plot_histogram(
    [ideal_counts, noisy_counts, perturbed_counts],
    legend=["Ideal", "Noisy", "Noisy with Perturbation"],
    title="Ideal vs Noisy vs Noisy with Perturbation",
    bar_labels=False
)
# Adjust layout to fit legend
plt.tight_layout()
plt.show()

def calculate_loss(ideal_counts, noisy_counts):
    total_shots = sum(ideal_counts.values())
    loss_percentages = []

    for state, ideal_count in ideal_counts.items():
        ideal_proportion = ideal_count / total_shots
        noisy_count = noisy_counts.get(state, 0)
        noisy_proportion = noisy_count / total_shots
        loss = abs(ideal_proportion - noisy_proportion) * 100
        loss_percentages.append(loss)

    return np.mean(loss_percentages)  # Average loss percentage


# Calculate average loss
noisy_loss = calculate_loss(ideal_counts, noisy_counts)
perturbed_loss = calculate_loss(ideal_counts, perturbed_counts)

# Print loss values
print(f"Noisy Loss: {noisy_loss}%")
print(f"Perturbed Loss: {perturbed_loss}%")

# Plot results
loss_values = [0, noisy_loss, perturbed_loss]  # Ideal loss is 0%
print(loss_values)
labels = ["Ideal", "Noisy", "Noisy with Perturbation"]

plt.bar(labels, loss_values, color=['blue', 'red', 'green'])
plt.title("Average Loss % Compared to Ideal")
plt.ylabel("Loss Percentage")
plt.ylim(0, max(loss_values) + 5)
plt.show()