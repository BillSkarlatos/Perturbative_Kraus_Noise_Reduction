# Quantum Internet noise handling as Perturbations

In this Project we attempt to simulate noise in an intermediate-to-large scale Quantum Internet application and handle it using $n^{th}$ order Quantum Hamiltionian time-dependent perturbations.

## Unperturbed Hamiltinian:

```math
\begin{equation*}
H_{0} = -i
\begin{pmatrix}
0 & 1  \\
1 & 0 
\end{pmatrix}
\end{equation*}
```
Hermitian operator that governs the unperturbed dynamics. Physically, this could correspond to the interaction-free evolution of a two-level quantum system (like a qubit).

The evolution under $H_{0}$​ alone would follow:

```math
\frac{d}{dt}\psi {t} = H_{0} \psi {t}
```

## Perturbation Example:

We used two time-dependent perturbation Hamiltonians were defined as:

```math
\begin{equation*}
V_{1}(t) = -i
\begin{pmatrix}
0 & 1  \\
-1 & 0 
\end{pmatrix}
\end{equation*}
```
```math
\begin{equation*}
V_{2}(t) = -i
\begin{pmatrix}
0 & i  \\
-i & 0 
\end{pmatrix}
\end{equation*}
```
