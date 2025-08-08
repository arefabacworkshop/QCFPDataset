from qiskit import QuantumCircuit, QuantumRegister
import numpy as np


def quantum_fourier_transform(circuit: QuantumCircuit, clock: QuantumRegister, n):
    circuit.swap(clock[0], clock[1])
    circuit.h(clock[0])
    # for j in reversed(range(n)):
    #     for k in reversed(range(j+1, n)):
            # circuit.cp(np.pi/float(2**(k-j)), clock[k], clock[j]);
    circuit.cp(np.pi/2, clock[1], clock[0]);
    circuit.h(clock[1])


def inverse_quantum_fourier_transform(circuit: QuantumCircuit, clock: QuantumRegister, n):
    circuit.h(clock[1])
    # for j in reversed(range(n)):
    #     for k in reversed(range(j+1, n)):
    #         circuit.cp(-np.pi/float(2**(k-j)), clock[k], clock[j])
    circuit.cp(-np.pi/2, clock[1], clock[0])
    circuit.h(clock[0])
    circuit.swap(clock[0], clock[1])