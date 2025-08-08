#!/usr/bin/env python3

from qiskit import QuantumCircuit

basic_fourier = QuantumCircuit(2, 2)
basic_fourier.h(0)
basic_fourier.cp(3.14 / 2, 0, 1)
basic_fourier.h(1)
basic_fourier.measure([0, 1], [0, 1])
