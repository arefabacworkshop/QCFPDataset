# from qiskit import Aer, execute
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp
import qiskit_nature
from qiskit_algorithms.minimum_eigensolvers import VQE, SamplingVQE, QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import QuantumInstance

import numpy as np
import random

N = 5  # Example protein length
num_states = 19
num_qubits = N * num_states

# Example coefficients (a_ij and b_ijkl)
a_ij_values = np.random.rand(N, num_states)  
b_ijkl_values = np.random.rand(N, num_states, N, num_states)  

# Define linear terms of the Hamiltonian
# H_linear = sum(PauliSumOp.from_list([(f'Z{i}', a_ij_values.flatten()[i])]) for i in range(num_qubits))
# linear_terms = [(f'Z{i}', a_ij_values.flatten()[i]) for i in range(num_qubits)]
linear_terms = [("I" * i + "Z" + "I" * (num_qubits - i - 1), a_ij_values.flatten()[i]) for i in range(num_qubits)]
H_linear = SparsePauliOp.from_list(linear_terms)

# Define quadratic terms of the Hamiltonian
# H_quadratic = sum(
#     PauliSumOp.from_list([(f'Z{i} Z{j}', b_ijkl_values.flatten()[k])])
#     for k, (i, j) in enumerate(np.ndindex(num_qubits, num_qubits))
# )
# quadratic_terms = [(f'Z{i} Z{j}', b_ijkl_values.flatten()[k])
#                    for k, (i, j) in enumerate(np.ndindex(num_qubits, num_qubits))]
# H_quadratic = SparsePauliOp.from_list(quadratic_terms)

# quadratic_terms = [("I" * i + "Z" + "I" * (j - i - 1) + "Z" + "I" * (num_qubits - j - 1), b_ijkl_values.flatten()[k])
#                    for k, (i, j) in enumerate(np.ndindex(num_qubits, num_qubits))]
# H_quadratic = SparsePauliOp.from_list(quadratic_terms)

quadratic_terms = [("I" * i + "Z" + "I" * (j - i - 1) + "Z" + "I" * (num_qubits - j - 1), b_ijkl_values.flatten()[k])
                   for k, (i, j) in enumerate(np.ndindex(N, num_states))]
H_quadratic = SparsePauliOp.from_list(quadratic_terms)

H = H_linear + H_quadratic

qc = QuantumCircuit(num_qubits)
params = [Parameter(f'theta_{i}') for i in range(num_qubits)]
for i in range(num_qubits):
    qc.ry(params[i], i)
for i in range(num_qubits - 1):
    qc.cx(i, i + 1)

# quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

optimizer = COBYLA()
# Define the VQE instance
vqe = VQE(ansatz=qc, optimizer=optimizer, quantum_instance=AerSimulator())
# vqe = SamplingVQE(ansatz=qc, optimizer=optimizer, quantum_instance=quantum_instance)
# vqe = SamplingVQE(ansatz=qc, optimizer=optimizer)
result = vqe.compute_minimum_eigenvalue(operator=H)

optimal_params = result.optimal_point
optimal_energy = result.eigenvalue.real

print("Optimal Parameters:", optimal_params)
print("Optimal Energy:", optimal_energy)

# Initial variant and its energy
current_variant = np.random.choice([0, 1], size=(N, num_states))  # Random initial variant
current_energy = optimal_energy  # Initial energy from VQE

# Metropolis criterion parameters
k_B = 1.38064852e-23  
T = 298.15  
num_iterations = 1000

def compute_energy(variant):
    return optimal_energy  # Placeholder for actual energy computation

for iteration in range(num_iterations):
    new_variant = current_variant.copy()
    i, j = np.random.randint(N), np.random.randint(num_states)
    new_variant[i, j] = 1 - new_variant[i, j]
    new_energy = compute_energy(new_variant)
    
    delta_E = new_energy - current_energy
    
    # Metropolis criterion
    if delta_E <= 0 or random.uniform(0, 1) < np.exp(-delta_E / (k_B * T)):
        current_variant = new_variant
        current_energy = new_energy

print("Optimal Protein Variant:")
print(current_variant)