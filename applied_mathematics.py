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