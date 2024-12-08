import matplotlib.pyplot as plt
from qiskit_dynamics.perturbation import solve_lmde_perturbation
import numpy as np

# Define time-independent unperturbed generator
def unperturbed_generator(t):
    return -1j * np.array([[0, 1], [1, 0]])

# Define perturbation components as callable functions
def perturbation_1(t):
    return -1j * np.array([[0, 1], [-1, 0]])

def perturbation_2(t):
    return -1j * np.array([[0, 1j], [-1j, 0]])

perturbations = [perturbation_1, perturbation_2]

# Specify initial state
initial_state = np.array([[1], [0]])

# Define the time span for the simulation
t_span = (0, 10)  # From time 0 to time 10

# Solve for the Dyson series terms up to order 2
result = solve_lmde_perturbation(
    generator=unperturbed_generator,
    perturbations=perturbations,
    expansion_order=2,
    t_span=t_span,
    y0=initial_state,
    dyson_in_frame=False,  # Ensures compatibility with y0
    expansion_method="dyson"
)

# Explore the structure of `result`
print("Result Structure:")
for label, matrix in result.items():
    print(f"Label: {label}, Type: {type(matrix)}, Value: {matrix}")

# Process only numerical data from `result`
term_labels = []
term_norms = []

for label, matrix in result.items():
    # Ensure the matrix is numerical before computing the norm
    if isinstance(matrix, np.ndarray):  # Check if `matrix` is a NumPy array
        term_labels.append(str(label))  # Convert label to string
        term_norms.append(np.linalg.norm(matrix))  # Compute Frobenius norm
    else:
        print(f"Skipping non-numerical entry for label {label}: {matrix}")

# Plot only valid numerical data
if term_labels and term_norms:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(term_labels, term_norms, color='skyblue')
    plt.xlabel("Perturbation Terms")
    plt.ylabel("Norm")
    plt.title("Norms of Dyson Series Perturbation Terms")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No valid numerical data to visualize.")

