from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import numpy as np

class QuantumCircuitGenerator:
    def __init__(self):
        # Noise parameters
        self.p1 = 0.1  # 1-qubit gate error
        self.p2 = 2 * self.p1  # 2-qubit gate error

        self.t1 = 50e3  # T1 time in nanoseconds
        self.t2 = 70e3  # T2 time in nanoseconds
        self.gate_time_1q = 50  # 1-qubit gate time in nanoseconds
        self.gate_time_2q = 150  # 2-qubit gate time in nanoseconds

        # Predefined noise model
        self.noise_model = self._create_noise_model()

    def _create_noise_model(self):
        """Create a noise model with depolarizing and thermal relaxation errors."""
        dep_error_1q = depolarizing_error(self.p1, 1)
        dep_error_2q = depolarizing_error(self.p2, 2)

        therm_error_1q = thermal_relaxation_error(self.t1, self.t2, self.gate_time_1q)
        therm_error_2q = thermal_relaxation_error(self.t1, self.t2, self.gate_time_2q).tensor(
            thermal_relaxation_error(self.t1, self.t2, self.gate_time_2q)
        )
        therm_error_3q = thermal_relaxation_error(self.t1, self.t2, self.gate_time_2q).tensor(
            thermal_relaxation_error(self.t1, self.t2, self.gate_time_2q).tensor(
                thermal_relaxation_error(self.t1, self.t2, self.gate_time_2q)
            )
        )
        dep_error_3q = depolarizing_error(self.p2, 3)
        error_3q = dep_error_3q.compose(therm_error_3q)

        error_1q = dep_error_1q.compose(therm_error_1q)
        error_2q = dep_error_2q.compose(therm_error_2q)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz', 'x', 'y'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        noise_model.add_all_qubit_quantum_error(error_3q, ['ccx'])

        return noise_model

    def generate_circuit(self, num_qubits):
        """Generate a quantum circuit with specified number of qubits."""
        if num_qubits < 2 or num_qubits > 7:
            raise ValueError("Number of qubits must be between 2 and 7.")

        qc = QuantumCircuit(num_qubits, num_qubits)

        # Add Hadamard gates to create superposition
        for qubit in range(num_qubits):
            qc.h(qubit)

        # Add entanglement using CNOT gates
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)

        # Add parameterized rotations for complexity
        theta = np.pi / (num_qubits + 1)
        for qubit in range(num_qubits):
            qc.rx(theta, qubit)
            qc.ry(theta / 2, qubit)

        # Add CCZ gates if number of qubits permits
        if num_qubits >= 3:
            qc.h(num_qubits - 1)
            qc.ccx(0, 1, num_qubits - 1)
            qc.h(num_qubits - 1)

        # Add measurements
        qc.measure(range(num_qubits), range(num_qubits))

        return qc

    def generate_simple_circuit(self, num_qubits):
        """Generate a simple quantum circuit with specified number of qubits."""
        if num_qubits < 2 or num_qubits > 7:
            raise ValueError("Number of qubits must be between 2 and 7.")

        qc = QuantumCircuit(num_qubits, num_qubits)

        # Add Hadamard gates to create superposition
        for qubit in range(num_qubits):
            qc.h(qubit)

        # Add measurements
        qc.measure(range(num_qubits), range(num_qubits))

        return qc

    def get_circuits_and_noise_models(self):
        """Generate circuits and noise models for qubit numbers from 2 to 7."""
        circuits_and_noise = {}
        for qubits in range(2, 8):
            circuit = self.generate_circuit(qubits)
            circuits_and_noise[qubits] = (circuit, self.noise_model)

        return circuits_and_noise

# Example usage
if __name__ == "__main__":
    generator = QuantumCircuitGenerator()
    circuits_and_noise = generator.get_circuits_and_noise_models()

    for qubits, (circuit, noise_model) in circuits_and_noise.items():
        print(f"\nQuantum Circuit for {qubits} qubits:")
        print(circuit)
        print(f"Noise Model for {qubits} qubits:")
        print(noise_model)
