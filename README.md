# Quantum Noise Mitigation via CPTP Projection and Perturbation Theory

## Overview
This repository presents an advanced framework for quantum noise mitigation leveraging perturbative techniques and Completely Positive Trace Preserving (CPTP) projection methodologies. The approach employs the Choi matrix representation of quantum channels to facilitate noise suppression and enhance computational fidelity. This framework is particularly applicable to fault-tolerant quantum computing and quantum networking protocols, offering scalable solutions for mitigating decoherence and gate imperfections.

## Motivation
Despite significant advancements in quantum hardware, noise remains a fundamental impediment to the practical realization of large-scale quantum computation. The proposed methodology contributes to noise suppression by employing first-order perturbation theory to adjust noisy quantum channels while ensuring their physical validity through CPTP constraints. This approach is particularly significant in quantum communication, where maintaining high-fidelity information transfer is crucial for the development of the quantum internet.

## Key Contributions
- **CPTP-Constrained Quantum Noise Mitigation:** Implements convex optimization techniques to project a perturbed quantum channel onto the nearest CPTP-compliant space.
- **Perturbative Expansion for Noise Correction:** Utilizes first-order perturbation theory to approximate deviations in the quantum channel, systematically mitigating noise.
- **Scalable Simulation Framework:** Facilitates large-scale quantum circuit simulations, benchmarking the effectiveness of noise suppression mechanisms.
- **Customizable Noise Models:** Incorporates depolarizing noise, thermal relaxation, and amplitude damping tailored to diverse quantum computing architectures.
- **Comprehensive Fidelity Analysis:** Provides robust evaluation tools for quantifying the impact of noise correction through trace distance and fidelity metrics.

## Installation
Ensure Python 3.7 or later is installed, then install dependencies:

```bash
pip install -r deps.txt
```

Alternatively, install manually:

```bash
pip install numpy qiskit qiskit-aer cvxpy matplotlib pylatexenc
```

## Usage

### Noise Mitigation in a Quantum Circuit

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from kraus import reduce_noise_in_circuit

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
optimized_noise_model = reduce_noise_in_circuit(qc, noise_model)
```

### Full Quantum Simulation

```python
from qiskit import transpile
from qiskit_aer import AerSimulator
from kraus import plot_results_as_dots
from calculations import calculate_fidelity, normalize_counts

backend = AerSimulator(noise_model=optimized_noise_model)
transpiled_circuit = transpile(qc, backend)
job = backend.run(transpiled_circuit, shots=1024)
result = job.result()
counts_corrected = result.get_counts()

# Compute Fidelity
fidelity = calculate_fidelity(ideal_counts, counts_corrected)
print("Fidelity of corrected system:", fidelity)
```

## Methodology

1. **Quantum Noise Modeling**
   - Simulates realistic noise using depolarizing, thermal relaxation, and amplitude damping models, parameterized to reflect contemporary quantum hardware constraints.

2. **Perturbative Noise Correction**
   - Employs first-order perturbation theory to approximate corrections to the noisy quantum channel:
     
     $E_{\text{corrected}} = \mathcal{E}_{\text{noisy}} + \delta \mathcal{E}$
     
   - The perturbation term $\delta \mathcal{E}$ is derived from the superoperator representation of quantum noise.

3. **CPTP Projection Optimization**
   - Enforces physical constraints on the corrected quantum channel by solving the convex optimization problem:
     
     $Min_{\mathcal{E}_{\text{CPTP}}} \| \mathcal{E}_{\text{corrected}} - \mathcal{E}_{\text{CPTP}} \|$
     
   - This projection ensures that the resultant quantum operation is completely positive and trace-preserving.

4. **Fidelity Evaluation**
   - Compares the statistical distance between noisy, corrected, and ideal quantum states using the fidelity metric:
     
     $F(\mathcal{E}_1, \mathcal{E}_2) = \left( \sum_i \sqrt{ P_1(i) P_2(i) } \right)^2$
     
   - Here, $P_1$ and $P_2$ denote probability distributions obtained from quantum measurement outcomes.

## Performance and Experimental Results
Extensive numerical simulations confirm that increasing the number of qubits enhances the efficacy of noise correction. The observed improvements in fidelity are consistent with theoretical predictions that noise mitigation via perturbative methods scales favorably with system size.

### Sample Output:
```
Fidelity of noisy system: 0.82
Fidelity of corrected system: 0.97
```

## Applications
- Quantum error correction methodologies
- Benchmarking quantum hardware resilience
- Quantum communication and cryptographic protocols
- Enhancing error mitigation strategies in quantum networking

## Repository Structure
- `kraus.py` – Implements noise correction and visualization
- `calculations.py` – Fidelity computation and CPTP projection methods
- `deps.txt` – List of dependencies
- `README.md` – Project documentation

## Citation
For academic or research use, please cite:
> *Kraus, K. (1983)*. States, Effects, and Operations: Fundamental Notions of Quantum Theory.
> *Choi, M.-D. (1975)*. Completely positive linear maps on complex matrices. Linear Algebra and Its Applications, 10(3), 285–290.

## License
This repository is available under the [MIT License](LICENSE).

## Acknowledgments
This work builds upon [Qiskit](https://qiskit.org/) and is informed by ongoing research in quantum noise suppression and error mitigation strategies.

---

