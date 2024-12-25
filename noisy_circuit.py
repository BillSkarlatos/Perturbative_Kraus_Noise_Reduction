from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

"""
COLLAPSES AFTER A CERTAIN AMMOUNT OF DATA, NOT STABLE
"""



# Function to calculate loss
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

# Define the base error rates
basic_error = 0.05
qubit_range = range(2, 12)
shots = 1024
perturbation_angle = basic_error * 0.01  # Adaptive perturbation # Small perturbation in radian

# Create the noise model (shared for both noisy and perturbed simulations)
noise_model = NoiseModel()
error_1q = depolarizing_error(basic_error, 1)  # depolarizing noise for single qubit gates
error_2q = depolarizing_error(1.5 * basic_error, 2)  # depolarizing noise for 2-qubit gates
noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Define simulators (shared noise model)
ideal_simulator = AerSimulator()
noisy_simulator = AerSimulator(noise_model=noise_model)
perturbed_simulator = AerSimulator(noise_model=noise_model)  # Same noise as noisy_simulator

all_noisy_losses = []
all_perturbed_losses = []

# Refined targeted perturbation function
def apply_selective_perturbation(circuit, epsilon):
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = instruction.qubits
        if gate.name == "cx":  # Target only 'cx' gates
            control = qubits[0]
            target = qubits[1]
            circuit.rz(epsilon, control._index)  # Use _index for Qubit objects
            circuit.rz(-epsilon, target._index)  # Use _index for Qubit objects


# Adjusted perturbation magnitude
perturbation_angle = 0.005  # Smaller perturbation in radians

# Simulation loop
all_noisy_losses = []
all_perturbed_losses = []

for n in qubit_range:
    # Define the quantum circuit dynamically for n qubits
    qc = QuantumCircuit(n)

    # Add some generic gates to the circuit
    for i in range(n):
        qc.rx(0.4, i)
        qc.ry(0.4, i)
        if i < n - 1:
            qc.cx(i, i + 1)

    # Transpile the circuit for the ideal simulation
    ideal_circuit = qc.copy()
    ideal_circuit.measure_all()
    ideal_circuit = transpile(ideal_circuit, ideal_simulator)

    # Transpile the circuit for the noisy simulation
    noisy_circuit = qc.copy()
    noisy_circuit.measure_all()
    noisy_circuit = transpile(noisy_circuit, noisy_simulator)

    # Apply selective perturbation to the perturbed circuit
    perturbed_circuit = qc.copy()  # Start with the original circuit without measurement gates
    apply_selective_perturbation(perturbed_circuit, perturbation_angle)
    perturbed_circuit.measure_all()  # Add measurement gates after applying perturbation
    perturbed_circuit = transpile(perturbed_circuit, perturbed_simulator)

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

    all_noisy_losses.append(noisy_loss)
    all_perturbed_losses.append(perturbed_loss)

# Print average losses and percentage difference
avg_noisy_loss = np.mean(all_noisy_losses)
avg_perturbed_loss = np.mean(all_perturbed_losses)
percentage_difference = ((avg_perturbed_loss - avg_noisy_loss) / avg_noisy_loss) * 100
print(f"Noisy Losses: {all_noisy_losses}")
print(f"Perturbed Losses: {all_perturbed_losses}")
print(f"Average Noisy Loss: {avg_noisy_loss:.2f}%")
print(f"Average Perturbed Loss: {avg_perturbed_loss:.2f}%")
print(f"Percentage Difference: {percentage_difference:.3f}%")



# Plot results
plt.plot(qubit_range, all_noisy_losses, label="Noisy Loss", marker='o')
plt.plot(qubit_range, all_perturbed_losses, label="Perturbed Loss", marker='x')
plt.title("Loss Percentages vs Number of Qubits")
plt.xlabel("Number of Qubits")
plt.ylabel("Loss Percentage")
plt.legend()
plt.grid()
plt.show()

# Print average losses
avg_noisy_loss = np.mean(all_noisy_losses)
avg_perturbed_loss = np.mean(all_perturbed_losses)
print(f"Noisy Losses: {all_noisy_losses}")
print(f"Perturbed Losses: {all_perturbed_losses}")
print(f"Average Noisy Loss: {avg_noisy_loss:.2f}%")
print(f"Average Perturbed Loss: {avg_perturbed_loss:.2f}%")
print(f"Percentage difference: {abs(avg_noisy_loss-avg_perturbed_loss)/avg_noisy_loss*100:.3f}%")

