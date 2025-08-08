import numpy as np
from numpy.random import randint
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from matplotlib import pyplot as plt


def bb84_encode(bits, basis):
    """
    Quantum-encode a set of bits onto a set of bases:
    If the basis is 0, a qubit in the Z-basis (i.e. |1> or |0>) is prepared,
    and if the basis is 1, in the X-basis (i.e. |-> or |+>)
    """
    code = []
    n = len(basis)
    for i in range(n):
        qc = QuantumCircuit(1, 1)
        if basis[i]:
            if bits[i]:
                qc.x(0)
            qc.h(0)
        else:
            if bits[i]:
                qc.h(0)
        qc.barrier()
        code.append(qc)
    return code


def measure_key(message, basis, backend=Aer.get_backend('aer_simulator')):
    """
    """
    measurements = []
    n = len(basis)
    for q in range(n):
        if basis[q]:
            message[q].h(0)
        message[q].measure(0, 0)
        result = backend.run(message[q], shots=1, memory=True).result()
        measured_bit = int(result.get_memory()[0])
        measurements.append(measured_bit)
    return measurements
        

def sync_results(bits, basis_a, basis_b):
    good_bits = []


def print_qbit(code, index, bits, bases, show=True):
    print(f'bit   = {bits[index]:2d}')
    print(f'basis = {bases[index]:2d}\n')
    if show:
        code[index].draw("mpl")
        plt.show()


if __name__ == "__main__":
    # set random seed and key length
    np.random.seed(seed=42)
    n = 10

    # Alice generates random bits and corresponding bases
    alice_bbits = randint(2, size=n)
    alice_bases = randint(2, size=n)

    # Alice encodes the bits in either
    #    Z basis (on Bloch sphere 0,0,1 or 0,0,-1)
    # or X basis (on Bloch sphere 1,0,0 or -1,0,0)
    key_msg = bb84_encode(alice_bbits, alice_bases)
    print_qbit(key_msg, 0, alice_bbits, alice_bases)
    print_qbit(key_msg, 1, alice_bbits, alice_bases)
    
    
    # Bob also choose random bases
    bob_bases = randint(2, size=n)
    bob_measr = measure_key(key_msg, bob_bases)
    print_qbit(key_msg, 0, bob_measr, bob_bases)
    print_qbit(key_msg, 1, bob_measr, bob_bases)
