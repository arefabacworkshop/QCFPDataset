from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

# Erstellen des Quantenschaltkreises
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)

# Visualisieren des Schaltkreises
print("Quantenschaltkreis:")
print(qc)

# Alternativ: Zeichnen des Schaltkreises
qc.draw()

# Manuelle Simulation des Schaltkreises
# Beispiel: Berechnung der Wahrscheinlichkeitsverteilung der Ergebnisse

# Erstellen eines Zustandsvektors
import numpy as np
statevector = np.zeros(2 ** qc.num_qubits)
statevector[0] = 1  # Anfangszustand ist |00>

# Anwenden der Quantengatter auf den Zustandsvektor
for gate in qc.data:
    gate_name = gate[0].name
    if gate_name == 'h':
        # Anwendung des Hadamard-Gatters
        qubit_index = qc.qubits.index(gate[1][0])
        statevector = np.dot(np.kron(np.eye(2 ** qubit_index), np.kron([[1, 1], [1, -1]], np.eye(2 ** (qc.num_qubits - qubit_index - 1)))), statevector)
    elif gate_name == 'cx':
        # Anwendung des CNOT-Gatters
        control_index = qc.qubits.index(gate[1][0])
        target_index = qc.qubits.index(gate[1][1])
        CNOT = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])
        statevector = np.dot(np.kron(np.eye(2 ** control_index), np.kron(CNOT, np.eye(2 ** (qc.num_qubits - control_index - 2)))), statevector)

# Berechnen der Wahrscheinlichkeitsverteilung
probabilities = np.abs(statevector) ** 2
counts = {}
for i, probability in enumerate(probabilities):
    binary_str = format(i, '0' + str(qc.num_qubits) + 'b')
    counts[binary_str] = probability

# Anzeigen der Wahrscheinlichkeitsverteilung
print("Wahrscheinlichkeitsverteilung:")
print(counts)
plot_histogram(counts)
