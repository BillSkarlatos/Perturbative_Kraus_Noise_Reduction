# Quantum Noise Correction with CPTP Projection

This repository provides tools for quantum noise correction in quantum circuits using perturbative methods and CPTP (Completely Positive Trace Preserving) projection. The implementation ensures physically consistent noise correction by leveraging the Choi matrix representation of quantum channels.

>This is a project under development and constant improvement and should be viewed and used as such.

## Features

- **CPTP Projection**: Ensures the noise correction process results in a valid quantum channel by projecting to the nearest CPTP matrix.
- **Perturbative Noise Correction**: Applies first-order corrections to noisy quantum circuits using superoperator representations.
- **Simulation and Analysis**: Compares noisy, corrected, and ideal outcomes to evaluate the performance of the correction.

---

## Installation

1. Install Python 3.7 or later.
2. Install the required dependencies:
   ```bash
   pip install qiskit qiskit-aer numpy
   ```
   or
   ```bash
   pip install -r deps.txt
   ```

---

## Usage

### Key Functions

1. **`project_to_cptp(choi)`**  
   Projects a Choi matrix to the closest CPTP matrix by ensuring non-negative eigenvalues.  
   **Parameters**:  
   - `choi`: A `Choi` object representing the channel to be projected.  
   **Returns**:  
   - A `Choi` object that is CPTP.

2. **`apply_noise_correction(qc, noise_model)`**  
   Corrects noise in a quantum circuit using perturbative methods and evaluates its performance.  
   **Parameters**:  
   - `qc`: A `QuantumCircuit` object representing the circuit to be corrected.  
   - `noise_model`: A `NoiseModel` object representing the noise in the quantum system.  
   **Returns**:  
   - `noisy_counts`: Counts from the noisy simulation.  
   - `corrected_counts`: Counts after applying noise correction.  
   - `ideal_counts`: Counts from the ideal (noise-free) simulation.  
   - `differences`: Differences in probabilities between ideal and corrected results.

### Example

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import numpy as np

# Define a 4-qubit quantum circuit
qc = QuantumCircuit(4, 4)
for qubit in range(4):
    qc.h(qubit)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.measure(range(4), range(4))

# Define a noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.1, 1), ['h'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.2, 2), ['cx'])

# Apply noise correction
from noise_correction import apply_noise_correction
noisy_counts, corrected_counts, ideal_counts, differences = apply_noise_correction(qc, noise_model)

# Display results
print("Noisy Counts:", noisy_counts)
print("Corrected Counts:", corrected_counts)
print("Ideal Counts:", ideal_counts)
print("Differences:", differences)
```

---

## How It Works

### 1. Noise Model
The noise model simulates errors like depolarizing and thermal relaxation, which occur in real quantum devices.

### 2. Perturbative Correction
The noise superoperator is calculated, and a correction matrix is derived using first-order perturbation theory:
\[ \text{Correction Matrix} = I - (\text{Noise Superoperator} - I) \]

### 3. CPTP Projection
If the correction matrix does not satisfy CPTP properties, it is projected to the nearest CPTP matrix using eigenvalue adjustments.

---

## Requirements

- Python >= 3.7
- [Qiskit](https://qiskit.org/)
- NumPy

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- Developed using [Qiskit](https://qiskit.org/).
- Inspired by advanced techniques in quantum error correction and noise modeling.

---
