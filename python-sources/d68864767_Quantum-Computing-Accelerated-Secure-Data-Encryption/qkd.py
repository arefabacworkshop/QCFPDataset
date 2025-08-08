```python
# Import necessary libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.visualization import plot_histogram
from quantum_computing import create_quantum_circuit, add_quantum_gates, simulate_quantum_circuit, run_quantum_circuit

# Define a function to create a BB84 protocol
def bb84_protocol(num_qubits):
    # Create a quantum circuit
    qc = create_quantum_circuit(num_qubits)

    # Alice prepares qubits in a state of her choice
    for i in range(num_qubits):
        qc.h(i)  # Apply Hadamard gate to put qubits in superposition

    # Alice sends these qubits to Bob
    # Bob measures the qubits in a basis of his choice
    for i in range(num_qubits):
        qc.measure(i, i)

    return qc

# Define a function to simulate the BB84 protocol
def simulate_bb84_protocol(num_qubits):
    # Create a BB84 protocol
    qc = bb84_protocol(num_qubits)

    # Simulate the quantum circuit
    counts = simulate_quantum_circuit(qc)

    # Print the results
    print("Simulation results: ", counts)

# Define a function to run the BB84 protocol on a real quantum computer
def run_bb84_protocol(num_qubits):
    # Create a BB84 protocol
    qc = bb84_protocol(num_qubits)

    # Run the quantum circuit
    counts = run_quantum_circuit(qc)

    # Print the results
    print("Real quantum computer results: ", counts)

# Test the BB84 protocol
if __name__ == "__main__":
    num_qubits = 4
    simulate_bb84_protocol(num_qubits)
    run_bb84_protocol(num_qubits)
```
