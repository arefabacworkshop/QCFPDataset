from qiskit import QuantumCircuit

def n_hadamard(n_qubits: int) -> QuantumCircuit:
    """Returns an n_hadamard gate"""
    circuit = QuantumCircuit(n_qubits)
    for i_qubit in range(n_qubits):
        circuit.h(i_qubit)
    return circuit
