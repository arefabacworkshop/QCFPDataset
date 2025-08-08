from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Function to create and run a quantum circuit
def create_and_run_circuit():
    # Create a Quantum Circuit acting on a quantum register of two qubits
    circuit = QuantumCircuit(2, 2)

    # Apply a Hadamard gate to qubit 0. This puts it into a superposition state.
    # |ψ⟩ = H|0⟩ = 1/√2(|0⟩ + |1⟩)
    circuit.h(0)

    # Apply a CNOT gate controlled by qubit 0 and targeted on qubit 1. This entangles them.
    # If qubit 0 is |1⟩, flip qubit 1, resulting in the entangled state 1/√2(|00⟩ + |11⟩)
    circuit.cx(0, 1)

    # Map the quantum measurement to the classical bits
    circuit.measure([0, 1], [0, 1])

    # Use Aer's qasm_simulator
    simulator = AerSimulator()

    # Execute the circuit on the qasm simulator
    result = simulator.run(circuit, shots=1000).result()

    # Returns counts
    counts = result.get_counts(circuit)
    print("\nTotal count for 00 and 11 are:", counts)

    # Plot a histogram
    plot_histogram(counts)

create_and_run_circuit()
