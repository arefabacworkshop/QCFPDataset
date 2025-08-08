import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UnitaryGate, QFT

def controlled_unitary_gate(unitary: UnitaryGate, exponent: int) -> QuantumCircuit:
    """
    Constructs a controlled gate for U^(exponent).
    """
    U_matrix = unitary.to_matrix()
    U_power = np.linalg.matrix_power(U_matrix, exponent)
    # Create the unitary gate for U^exponent and then control it.
    CU_gate = UnitaryGate(U_power, label=f"U^{exponent}").control(1)
    return CU_gate

def qpe_circuit(n_count: int, unitary: UnitaryGate) -> QuantumCircuit:
    """
    Constructs a Quantum Phase Estimation (QPE) circuit.
    
    Parameters:
      n_count: Number of qubits in the counting register.
      unitary: A UnitaryGate whose eigenphase we wish to estimate.
    
    The circuit uses (n_count + 1) qubits: n_count for phase estimation and 1 work qubit.
    The work register is initialized in the eigenstate |1> of U.
    Controlled operations U^(2^(n_count - j - 1)) are applied for each counting qubit.
    Then the inverse QFT is applied to the counting register, and the counting register is measured.
    """
    total_qubits = n_count + 1
    qc = QuantumCircuit(total_qubits, n_count)
    
    # 1. Apply Hadamard gates to the counting register.
    qc.h(range(n_count))
    
    # 2. Prepare the work register in the eigenstate |1>.
    qc.x(n_count)
    
    # 3. Apply controlled-U^(2^(n_count - j - 1)) for each counting qubit j.
    for j in range(n_count):
        exponent = 2 ** (n_count - j - 1)
        CU_gate = controlled_unitary_gate(unitary, exponent)
        qc.append(CU_gate, [j, n_count])
    
    # 4. Apply the inverse QFT on the counting register.
    qc.append(QFT(n_count, inverse=True, do_swaps=True), list(range(n_count)))
    
    # 5. Measure the counting register.
    qc.measure(range(n_count), range(n_count))
    
    return qc

# Define the eigenphase to be estimated.
phi = 5/16  # For example, phi = 0.3125
U_matrix = np.array([[1, 0], [0, np.exp(2 * np.pi * 1j * phi)]])
U_gate = UnitaryGate(U_matrix, label="U")

n_count = 4  # Number of counting qubits; increases resolution.

# Build the QPE circuit.
qpe_circ = qpe_circuit(n_count, U_gate)

# Save the circuit diagram.
os.makedirs("images", exist_ok=True)
qpe_circ.draw('mpl', filename="images/qpe_circuit.png")
plt.close()
print("Circuit saved to 'images/qpe_circuit.png'")

# Use local simulator.
simulator = Aer.get_backend('qasm_simulator')
transpiled_qpe = transpile(qpe_circ, backend=simulator)

# Run simulation.
result = simulator.run(transpiled_qpe, shots=1000).result()
counts = result.get_counts()
probabilities = {k: v/1000 for k, v in counts.items()}

# Plot and save the histogram of measurement outcomes.
fig = plot_histogram(probabilities, title="QPE Results")
fig.savefig("images/qpe_results.png", bbox_inches="tight")
print("Results saved to 'images/qpe_results.png'")
