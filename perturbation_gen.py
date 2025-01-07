from qiskit.quantum_info import SuperOp
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError

def get_noise_matrices(qc, noise_model):
    """
    Calculate and return the noise matrices (SuperOp) for each gate in the given noise model.

    Args:
        qc (QuantumCircuit): The quantum circuit to analyze.
        noise_model (NoiseModel): The noise model to apply.

    Returns:
        dict: A dictionary mapping gate names to their respective noise superoperators.
    """
    noise_matrices = {}

    # Extract noise channels and compute their SuperOp representation
    for gate_name in noise_model._default_quantum_errors.keys():
        error = noise_model._default_quantum_errors[gate_name]
        superop = SuperOp(error)  # Convert error channel to SuperOp
        noise_matrices[gate_name] = superop

    return noise_matrices