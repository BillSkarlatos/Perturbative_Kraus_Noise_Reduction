from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Define the quantum circuit (reconstructing our circuit from simple_circuit.py)
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

# Print the circuit
print(qc)

# Create a noise model
noise_model = NoiseModel()
error_1q = depolarizing_error(0.01, 1)  # 1% depolarizing noise for single-qubit gates
error_2q = depolarizing_error(0.02, 2)  # 2% depolarizing noise for two-qubit gates
noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# Configure the AerSimulator
simulator = AerSimulator(noise_model=noise_model)

# Transpile the circuit for the simulator
compiled_circuit = transpile(qc, simulator)

# Run the simulation
noisy_job = simulator.run(compiled_circuit, shots=1024)
noisy_result = noisy_job.result()
noisy_counts = noisy_result.get_counts()

# Plot the noisy results
plot_histogram(noisy_counts, title="Noisy Simulation Results")
plt.show()
