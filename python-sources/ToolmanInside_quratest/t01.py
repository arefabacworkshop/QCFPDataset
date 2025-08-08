from unittest import TestCase
from mutator import RandomMutator, QFTMutator, UCNOTMutator
from qiskit import QuantumCircuit, execute, Aer
from line import Circuit
from adder_test import adder

class TestAdder_01(TestCase):
    def test_0000_prob_0_percent(self):
        new_circuit = Circuit(4)
        mutate_circuit = RandomMutator().generate_circuit(new_circuit)
        result = adder(mutate_circuit.code)
        result = adder(new_circuit.code)
        self.assertTrue(len(result) >= 1)