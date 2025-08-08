import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Set problem size
n = 4  # Number of input bits
s1 = '1010'  # Secret string 1
s2 = '1100'  # Secret string 2

# Oracle that returns f(x) = [s1·x, s2·x]
def dual_bv_oracle(n, s1, s2):
    qr = QuantumRegister(n + 2)
    oracle = QuantumCircuit(qr)

    s1 = s1[::-1]
    s2 = s2[::-1]

    for i, bit in enumerate(s1):
        if bit == '1':
            oracle.cx(i, n)  # s1 dot x goes to ancilla 1 (qubit n)

    for i, bit in enumerate(s2):
        if bit == '1':
            oracle.cx(i, n+1)  # s2 dot x goes to ancilla 2 (qubit n+1)

    return oracle

# Dual BV Algorithm
def dual_bv_circuit(n, s1, s2):
    qr = QuantumRegister(n + 2)
    cr = ClassicalRegister(n + 2)
    qc = QuantumCircuit(qr, cr)

    # Step 1: Initialize ancilla qubits in |-> state
    qc.x(qr[n])
    qc.x(qr[n+1])
    qc.h(qr[n])
    qc.h(qr[n+1])

    # Step 2: Hadamard on input register
    for i in range(n):
        qc.h(qr[i])

    # Step 3: Apply oracle
    oracle = dual_bv_oracle(n, s1, s2)
    qc.compose(oracle, inplace=True)

    # Step 4: Hadamard again on input register
    for i in range(n):
        qc.h(qr[i])

    # Step 5: Measure input qubits
    for i in range(n):
        qc.measure(qr[i], cr[i])
    # Optional: measure ancilla
    qc.measure(qr[n], cr[n])
    qc.measure(qr[n+1], cr[n+1])

    return qc

# Run the circuit
qc = dual_bv_circuit(n, s1, s2)
qc.draw('mpl')  # Optional

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=1024).result()
counts = result.get_counts()

# Show results
print("Measurement counts:")
for bitstring, count in counts.items():
    print(f"{bitstring}: {count}")

# Extract measured input register bits
measured_inputs = [k[-n:] for k in counts.keys()]
print(f"\nRecovered s1 XOR s2 values (if unique): {set(measured_inputs)}")

plot_histogram(counts)
plt.show()