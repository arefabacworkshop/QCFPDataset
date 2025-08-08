from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
import numpy as np
import argparse
import sys

def rng(vals, shots=1):
    max_val = max(vals.keys())
    nbits = len(bin(max_val)) - 2
    qreg_q = QuantumRegister(nbits, 'q')
    qc = QuantumCircuit(qreg_q)
    
    def unitary_from_column(column_vector):
        n = len(column_vector)

        # Normalize the column vector
        u1 = column_vector / np.linalg.norm(column_vector)

        # Initialize the unitary matrix with the first column
        unitary_matrix = np.array([u1]).T

        # Construct the remaining columns using the Gram-Schmidt process
        for j in range(1, n):
            v = np.random.randn(n)  # Start with a random vector
            for i in range(j):
                proj = np.dot(unitary_matrix[:, i], v) * unitary_matrix[:, i]
                v = v - proj
            uj = v / np.linalg.norm(v)
            unitary_matrix = np.c_[unitary_matrix, uj]

        return unitary_matrix

    v = np.array([np.sqrt(vals.get(i, 0)) for i in range(2**nbits)])
    unitary = unitary_from_column(v)
    op = Operator(unitary)
    qc.append(op, qreg_q)
    qc.measure_all()
    aer = AerSimulator()
    counts = aer.run(qc, shots=shots).result().get_counts()
    return {int(k, 2): v/shots for k, v in counts.items()}

if __name__ == '__main__':

    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    arg2 = sys.argv[2] if len(sys.argv) > 2 else 1

    # Parse the input values dictionary
    vals = {}
    for pair in arg1.split(','):
        key, value = pair.split(':')
        vals[int(key.strip())] = float(value.strip())

    # Run the rng function
    result = rng(vals, int(arg2))
    
    if arg2 == 1:
        for k,v in result.items():
            if v == 1:
                print(k)
    else:
        print(result)