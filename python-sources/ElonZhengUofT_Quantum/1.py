from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import numpy as np

def gray_code(n):
    if n == 0:
        return ['']
    first_half = gray_code(n-1)
    second_half = first_half[::-1]
    first_half = ['0' + code for code in first_half]
    second_half = ['1' + code for code in second_half]
    return first_half + second_half

def calculate_rotation_angles(n):
    """Calculate rotation angles for a uniformly controlled rotation gate"""
    angles = []
    gray_codes = gray_code(n)
    for code in gray_codes:
        # Example calculation: angles based on the position of '1' bits
        angle = sum(int(bit) for bit in code) * np.pi / (2 ** n)
        angles.append(angle)
    return angles


def build_uniformly_controlled_rotation_circuit(control, target):
    """Build a quantum circuit implementing uniformly controlled rotations"""
    qc = QuantumCircuit(control + 1)
    angles = calculate_rotation_angles(control)

    gray_codes = gray_code(control)
    for idx, code in enumerate(gray_codes):
        for ctrl_qubit in range(control):
            if code[ctrl_qubit] == '1':
                qc.cx(ctrl_qubit, control)
        qc.ry(angles[idx], control)
        for ctrl_qubit in range(control):
            if code[ctrl_qubit] == '1':
                qc.cx(ctrl_qubit, control)

    return qc

# Example usage:
n_control_qubits = 3
target_qubit = 3  # In a circuit with 4 qubits, qubit index 3 is the 4th qubit
qc = build_uniformly_controlled_rotation_circuit(n_control_qubits, target_qubit)
print(qc)

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
qobj = assemble(compiled_circuit)
result = execute(compiled_circuit, backend=simulator).result()
counts = result.get_counts()
print(counts)
plot_histogram(counts).show()