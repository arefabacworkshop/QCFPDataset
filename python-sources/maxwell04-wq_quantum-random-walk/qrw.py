# import libraries
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from numpy import pi, array
import numpy as np
from helper import print_table_from_dict

# Define the coin
n = 3 # dimension of graph
n_d = int(np.ceil(np.log2(n))) # dimension of coin

# declare the shift operator
ed = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [1, 0, 0]])

S = np.zeros((2**(n+n_d), 2**(n+n_d)))
for d in range(2**n_d):
    for x in range(2**n):
        vd = np.zeros((1, 2**n_d))
        vd[0, d] = 1
        vx = np.zeros((1, 2**n), dtype=np.int64)
        vx[0, x] = 1
        ved = np.zeros((1, 2**n), dtype=np.int64)
        ed_decimal = int("".join(map(str, ed[d])), 2)
        ved[0, np.bitwise_xor(ed_decimal, x)] = 1
        col  = np.kron(ved, vd).reshape(2**(n+n_d), 1)
        row  = np.kron(vx, vd).reshape(1, 2**(n+n_d))
        S += np.dot(col, row)
np.set_printoptions(threshold=np.inf)
S_gate = UnitaryGate(S, label='shift\noperator')
print("Shift operator:\n", S_gate.params)

# Declare the coin operator
C = UnitaryGate(2/2**n_d*np.ones((2**n_d, 2**n_d)) - np.eye(2**n_d), label='coin')
print("Coin operator:\n", C.params)

# Create the quantum circuit 
qc  = QuantumCircuit()
steps = 3
# Add quantum registers to the circuit
shift = QuantumRegister(n, 's') # qubits associated with the graph nodes
coin = QuantumRegister(n_d, 'c') # qubit associated with the coin
qc.add_register(coin)
qc.add_register(shift)
# Create equal superposition of coin and walker
[qc.h(coin[i]) for i in range(n_d)]
qc.barrier()
for _ in range(steps):
    qc.append(C, coin)
    # Apply the walk
    qc.append(S_gate, range(qc.num_qubits))
    qc.barrier()
# Measure the state of the walker
qc.measure_all()
# Display the quantum circuit
qc.draw('mpl', filename=f"qrw_circuit_{steps}.png")

# Simulate the quantum circuit 
sim_ideal = AerSimulator()
tqc = transpile(qc, sim_ideal)
result_ideal = sim_ideal.run(tqc).result()
plot_histogram(result_ideal.get_counts(), filename=f"./images/qrw_sim_results_{steps}.png")

print_table_from_dict(result_ideal.get_counts(), n, n_d)

