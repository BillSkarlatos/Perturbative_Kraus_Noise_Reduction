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

basic_error=0.02

# Create noise model
noise_model = NoiseModel()
error_1q = depolarizing_error(basic_error, 1)  # depolarizing noise
error_2q = depolarizing_error(1.5*basic_error, 2)  # 1.5 x depolarizing noise
noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

perturbed_noise_model = NoiseModel()
perturbed_noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ['rx', 'ry', 'rz', 'h'])
perturbed_noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])

# Define simulators
ideal_simulator = AerSimulator()  # No noise
noisy_simulator = AerSimulator(noise_model=noise_model)  # With noise
perturbed_simulator = AerSimulator(noise_model=perturbed_noise_model)  # With perturbed noise

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

shot_range = range(1024, 10241, 1024)
noisy_losses = []
perturbed_losses = []

for shots in shot_range:
    # Transpile the circuit for the simulators
    ideal_circuit = transpile(qc, ideal_simulator)
    noisy_circuit = transpile(qc, noisy_simulator)
    perturbed_circuit = transpile(qc, perturbed_simulator)

    # Run ideal simulation
    ideal_job = ideal_simulator.run(ideal_circuit, shots=shots)
    ideal_result = ideal_job.result()
    ideal_counts = ideal_result.get_counts()

    # Run noisy simulation
    noisy_job = noisy_simulator.run(noisy_circuit, shots=shots)
    noisy_result = noisy_job.result()
    noisy_counts = noisy_result.get_counts()

    # Run perturbed simulation
    perturbed_job = perturbed_simulator.run(perturbed_circuit, shots=shots)
    perturbed_result = perturbed_job.result()
    perturbed_counts = perturbed_result.get_counts()

    # Calculate losses
    noisy_loss = calculate_loss(ideal_counts, noisy_counts)
    perturbed_loss = calculate_loss(ideal_counts, perturbed_counts)

    noisy_losses.append(noisy_loss)
    perturbed_losses.append(perturbed_loss)

# Calculate average losses
avg_noisy_loss = np.mean(noisy_losses)
avg_perturbed_loss = np.mean(perturbed_losses)

# Plot results
plt.plot(shot_range, noisy_losses, label="Noisy Loss", marker='o')
plt.plot(shot_range, perturbed_losses, label="Perturbed Loss", marker='x')
plt.title(f"Loss Percentages vs Shot Number with error rates at {basic_error*100:.1f}%, {basic_error*150:.1f}%")
plt.xlabel("Shot Number")
plt.ylabel("Loss Percentage")
plt.legend()
plt.grid()
plt.show()

# Print average losses under the graph
print(f"Average Noisy Loss: {avg_noisy_loss:.2f}%")
print(f"Average Perturbed Loss: {avg_perturbed_loss:.2f}%")
