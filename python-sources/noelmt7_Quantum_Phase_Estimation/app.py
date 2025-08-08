import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import numpy as np

# Define QFT dagger (inverse QFT)
def qft_dagger(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    for i in range(num_qubits):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), j, i)
        qc.h(i)
    return qc

# Define Phase Estimation circuit
def phase_estimation(unitary, eigenstate, num_ancillae):
    num_unitary_qubits = unitary.num_qubits
    qpe = QuantumCircuit(num_ancillae + num_unitary_qubits, num_ancillae)
    
    # Apply Hadamard to ancilla qubits
    for qubit in range(num_ancillae):
        qpe.h(qubit)
    
    # Add eigenstate to target qubits
    qpe.compose(eigenstate, qubits=range(num_ancillae, num_ancillae + num_unitary_qubits), inplace=True)

    # Controlled unitary operations
    for i in range(num_ancillae):
        repetitions = 2 ** i
        controlled_unitary = unitary.control(1)
        for _ in range(repetitions):
            qpe.append(controlled_unitary, [i] + list(range(num_ancillae, num_ancillae + num_unitary_qubits)))

    # Apply inverse QFT
    qft_inverse = QFT(num_ancillae, do_swaps=False).inverse()
    qpe.append(qft_inverse, range(num_ancillae))

    # Measure ancilla qubits
    qpe.measure(range(num_ancillae), range(num_ancillae))
    return qpe

# Streamlit interface
st.title("Quantum Phase Estimation Simulator")

# User input
st.sidebar.header("Input Parameters")
num_ancillae = st.sidebar.slider("Number of Ancilla Qubits", 1, 5, 3)
shots = st.sidebar.number_input("Number of Shots", value=1024, step=256)

# Define a unitary operator
unitary = QuantumCircuit(1)
unitary.x(0)
unitary.z(0)

# Define an eigenstate
eigenstate = QuantumCircuit(1)
eigenstate.h(0)

# Generate the QPE circuit
qpe_circuit = phase_estimation(unitary, eigenstate, num_ancillae)

# Display the circuit
st.write("### Quantum Phase Estimation Circuit:")
fig = qpe_circuit.draw(output='mpl')  # 'mpl' output format for matplotlib
st.pyplot(fig)

# Run the simulation
if st.button("Simulate"):
    st.write("Simulating...")
    simulator = AerSimulator()
    transpiled_circuit = transpile(qpe_circuit, simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts()

    # Display results
    st.write("### Measurement Results:")
    st.json(counts)

    # Save and display the histogram
    histogram_path = "histogram.png"
    plot_histogram(counts, title="Measurement Outcomes").savefig(histogram_path)
    st.write("### Measurement Outcome Histogram:")
    st.image(histogram_path)
