import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Define the Hamiltonian: H = Z0 + Z1 + Z0*Z1.
def hamiltonian_energy(bitstring: str) -> float:
    z0 = 1 if bitstring[0] == '0' else -1
    z1 = 1 if bitstring[1] == '0' else -1
    return z0 + z1 + (z0 * z1)

# Define a simple ansatz circuit for two qubits using two parameters.
def ansatz_circuit(params):
    theta, phi = params
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.ry(phi, 1)
    return qc

# Cost function for VQE, using statevector simulation.
def cost_function(params):
    qc = ansatz_circuit(params)
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc).result()
    state = np.asarray(result.get_statevector(qc))
    energy = 0.0
    basis_states = ['00', '01', '10', '11']
    for i, amplitude in enumerate(state):
        prob = np.abs(amplitude)**2
        energy += prob * hamiltonian_energy(basis_states[i])
    return energy

# Optimize parameters using a classical optimizer.
initial_params = [0.0, 0.0]
opt_result = minimize(cost_function, initial_params, method='COBYLA')
optimal_params = opt_result.x
optimal_energy = opt_result.fun

print("Optimal parameters:", optimal_params)
print("Optimal energy:", optimal_energy)

# Build the final ansatz circuit with the optimal parameters.
ansatz = ansatz_circuit(optimal_params)

# Create a new circuit with classical registers.
final_circuit = QuantumCircuit(2, 2)
final_circuit.compose(ansatz, inplace=True)
final_circuit.measure([0, 1], [1, 0])

# Save the circuit diagram.
os.makedirs("images", exist_ok=True)
final_circuit.draw('mpl', filename="images/vqe_circuit.png")
plt.close()
print("Circuit saved to 'images/vqe_circuit.png'")

# Run on the local qasm_simulator.
simulator = Aer.get_backend('qasm_simulator')
transpiled_circuit = transpile(final_circuit, backend=simulator)
result = simulator.run(transpiled_circuit, shots=1000).result()
counts = result.get_counts()

# Normalize counts to probabilities.
probabilities = {state: count/1000 for state, count in counts.items()}

# Plot and save the histogram.
fig = plot_histogram(probabilities, title="VQE Results")
fig.savefig("images/vqe_results.png", bbox_inches="tight")
print("Results saved to 'images/vqe_results.png'")
