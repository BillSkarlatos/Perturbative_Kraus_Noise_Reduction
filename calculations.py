import numpy as np
import cvxpy as cp

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

# Function to normalize data (counts)
def normalize_counts(counts, shots):
    """
    Normalize the counts data by dividing by the total number of shots to get probabilities.
    
    Args:
    - counts: A dictionary of measurement outcomes with their counts.
    - shots: Total number of shots (experiments).
    
    Returns:
    - normalized_counts: A dictionary of normalized probabilities.
    """
    normalized_counts = {k: v / shots for k, v in counts.items()}
    return normalized_counts


def calculate_fidelity(ideal_counts, test_counts):
    """
    Calculate the fidelity between the ideal and test distributions.

    Args:
    - ideal_counts (dict): A dictionary representing the ideal distribution (bitstring -> counts).
    - test_counts (dict): A dictionary representing the noisy or corrected distribution (bitstring -> counts).

    Returns:
    - float: The fidelity between the ideal and test distributions.
    """
    # Normalize the counts to probabilities
    ideal_total = sum(ideal_counts.values())
    test_total = sum(test_counts.values())

    ideal_probs = {key: value / ideal_total for key, value in ideal_counts.items()}
    test_probs = {key: value / test_total for key, value in test_counts.items()}

    # Calculate the fidelity
    fidelity = 0
    for key in ideal_probs:
        # If the key is missing in test_probs, assume the probability is 0
        p_ideal = ideal_probs.get(key, 0)
        p_test = test_probs.get(key, 0)
        fidelity += np.sqrt(p_ideal * p_test)

    fidelity = fidelity**2
    return fidelity

from qiskit import QuantumCircuit

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from qiskit import QuantumCircuit

def generate(num_qubits):
    """
    Generate a complex quantum circuit for a given number of qubits and create a corresponding noise model.

    Args:
    - num_qubits (int): Number of qubits in the circuit.

    Returns:
    - QuantumCircuit: A quantum circuit with the specified number of qubits.
    - NoiseModel: A noise model tailored for the circuit.
    """
    # Step 1: Create a quantum circuit
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Apply Hadamard gates to create superposition
    for qubit in range(num_qubits):
        qc.h(qubit)

    # Add entanglement using CNOT gates
    for qubit in range(num_qubits - 1):
        qc.cx(qubit, qubit + 1)

    # Add controlled gates (CZ and CH) for additional complexity
    for qubit in range(num_qubits - 1):
        qc.cz(qubit, qubit + 1)  # Controlled-Z
        if qubit + 2 < num_qubits:
            qc.ch(qubit, qubit + 2)  # Controlled-Hadamard

    # Add single-qubit rotations for randomness
    for qubit in range(num_qubits):
        qc.rx(0.5 * (qubit + 1), qubit)
        qc.ry(1.0 * (qubit + 1), qubit)
        qc.rz(1.5 * (qubit + 1), qubit)

    # Add barriers for clarity
    qc.barrier()

    # Apply additional long-range entanglement
    for qubit in range(num_qubits):
        if qubit + 2 < num_qubits:
            qc.cx(qubit, qubit + 2)

    # Measure all qubits
    for qubit in range(num_qubits):
        qc.measure(qubit, qubit)

    # Step 2: Create a noise model
    noise_model = NoiseModel()

    # Depolarizing error: 2% for 1-qubit gates, 5% for 2-qubit gates
    depol_error_1q = depolarizing_error(0.02, 1)
    depol_error_2q = depolarizing_error(0.05, 2)

    # Thermal relaxation error: realistic values for qubits
    thermal_error_1q = thermal_relaxation_error(t1=50e-6, t2=30e-6, time=20e-6)
    thermal_error_2q = thermal_relaxation_error(t1=50e-6, t2=30e-6, time=40e-6)

    # Amplitude damping error: 10% probability
    amp_damp_error = amplitude_damping_error(0.1)

    # Composite errors for single- and two-qubit gates
    composite_1q_error = depol_error_1q.compose(thermal_error_1q)
    composite_2q_error = depol_error_2q.compose(thermal_error_2q)

    # Add errors to the noise model
    # Single-qubit gates
    noise_model.add_all_qubit_quantum_error(composite_1q_error, ['u1', 'u2', 'u3', 'id'])
    noise_model.add_all_qubit_quantum_error(amp_damp_error, ['id'])

    # Two-qubit gates
    noise_model.add_all_qubit_quantum_error(composite_2q_error, ['cx'])

    return qc, noise_model
