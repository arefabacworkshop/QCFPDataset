import matplotlib.pyplot as plt
import numpy as np
import math
from qiskit.circuit.library import MCXGate

from qiskit import *
from qiskit_aer import QasmSimulator

# importing Qiskit
# from qiskit import Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools
from qiskit.visualization import plot_histogram

clause_list = [[0,1],
               [0,2],
               [1,3],
               [2,3]]

def XOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)

    
# We will use separate registers to name the bits
in_qubits = QuantumRegister(2, name='input')
out_qubit = QuantumRegister(1, name='output')
qc = QuantumCircuit(in_qubits, out_qubit)
XOR(qc, in_qubits[0], in_qubits[1], out_qubit)
qc.draw()

# Create separate registers to name bits
var_qubits = QuantumRegister(4, name='v')  # variable bits
clause_qubits = QuantumRegister(4, name='c')  # bits to store clause-checks

# Create quantum circuit
qc = QuantumCircuit(var_qubits, clause_qubits)

# Use XOR gate to check each clause
i = 0
for clause in clause_list:
    XOR(qc, clause[0], clause[1], clause_qubits[i])
    i += 1

qc.draw()


# Create separate registers to name bits
var_qubits = QuantumRegister(4, name='v')
clause_qubits = QuantumRegister(4, name='c')
output_qubit = QuantumRegister(1, name='out')
qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit)

# Compute clauses
i = 0
for clause in clause_list:
    XOR(qc, clause[0], clause[1], clause_qubits[i])
    i += 1

# Flip 'output' bit if all clauses are satisfied
# qc.ccx(clause_qubits[0], output_qubit)
qc.append(MCXGate(num_ctrl_qubits=len(clause_qubits)), clause_qubits[:] + [output_qubit[0]])

qc.draw()


var_qubits = QuantumRegister(4, name='v')
clause_qubits = QuantumRegister(4, name='c')
output_qubit = QuantumRegister(1, name='out')
cbits = ClassicalRegister(4, name='cbits')
qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit, cbits)

def sudoku_oracle(qc, clause_list, clause_qubits):
    # Compute clauses
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1

    # Flip 'output' bit if all clauses are satisfied
    # qc.mct(clause_qubits, output_qubit)
    qc.append(MCXGate(num_ctrl_qubits=len(clause_qubits)), clause_qubits[:] + [output_qubit[0]])


    # Uncompute clauses to reset clause-checking bits to 0
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1

sudoku_oracle(qc, clause_list, clause_qubits)
qc.draw()

from qiskit.circuit.library import MCXGate
from qiskit import QuantumCircuit

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    
    # Step 1: H on all qubits
    qc.h(range(nqubits))
    # Step 2: X on all qubits
    qc.x(range(nqubits))
    # Step 3: Multi-controlled Z gate using MCX
    qc.h(nqubits - 1)
    mcx = MCXGate(nqubits - 1)  # n-1 control qubits
    qc.append(mcx, list(range(nqubits)))
    qc.h(nqubits - 1)
    # Step 4: X on all qubits
    qc.x(range(nqubits))
    # Step 5: H on all qubits
    qc.h(range(nqubits))
    
    # Turn into a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s


var_qubits = QuantumRegister(4, name='v')
clause_qubits = QuantumRegister(4, name='c')
output_qubit = QuantumRegister(1, name='out')
cbits = ClassicalRegister(4, name='cbits')
qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit, cbits)

# Initialize 'out0' in state |->
qc.initialize([1, -1]/np.sqrt(2), output_qubit)

# Initialize qubits in state |s>
qc.h(var_qubits)
qc.barrier()  # for visual separation

## First Iteration
# Apply our oracle
sudoku_oracle(qc, clause_list, clause_qubits)
qc.barrier()  # for visual separation
# Apply our diffuser
qc.append(diffuser(4), [0,1,2,3])

## Second Iteration
sudoku_oracle(qc, clause_list, clause_qubits)
qc.barrier()  # for visual separation
# Apply our diffuser
qc.append(diffuser(4), [0,1,2,3])

# Measure the variable qubits
qc.measure(var_qubits, cbits)

qc.draw(fold=-1)

# Simulate and plot results

backend = QasmSimulator()

transpiled_qc = transpile(qc, backend)
result = backend.run(transpiled_qc.reverse_bits()).result()
plot_histogram(result.get_counts())
plt.show()