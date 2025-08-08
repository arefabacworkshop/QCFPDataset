import qiskit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

def apply_circuit_to_state(initial_state):
    # Quantum circuit for the transformation
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr)

    # Applying the sequence of CNOT gates
    qc.cx(0, 1)  # CNOT with qubit 0 as control and qubit 1 as target
    qc.cx(0, 4)  # CNOT with qubit 0 as control and qubit 4 as target
    qc.cx(2, 0)  # CNOT with qubit 2 as control and qubit 0 as target
    qc.cx(3, 0)  # CNOT with qubit 3 as control and qubit 0 as target

    # Apply the circuit to the initial state
    state = Statevector.from_label(initial_state)
    final_state = state.evolve(qc)
    return final_state

# Define the four states
psi_00 = '00000' # Placeholder for the actual state representation of psi_00
psi_01 = '10011' # Placeholder for the actual state representation of psi_01
psi_10 = '10101' # Placeholder for the actual state representation of psi_10
psi_11 = '01100' # Placeholder for the actual state representation of psi_11

# Apply the circuit to each state and get the resulting state
result_psi_00 = apply_circuit_to_state(psi_00)
result_psi_01 = apply_circuit_to_state(psi_01)
result_psi_10 = apply_circuit_to_state(psi_10)
result_psi_11 = apply_circuit_to_state(psi_11)

# Print the results
print("Resulting state for psi_00:", result_psi_00)
print("Resulting state for psi_01:", result_psi_01)
print("Resulting state for psi_10:", result_psi_10)
print("Resulting state for psi_11:", result_psi_11)
