import numpy as np
from qiskit import transpile
from utils import run_circuit

# Visualization of the problem graph
import matplotlib.pyplot as plt
import networkx as nx

# Matricial computation and calssical optimization
import numpy as np
from scipy.optimize import minimize

# Creation of quantum circuits
from qiskit import QuantumCircuit
from qiskit import QuantumRegister

# Structure used to build Hamiltonians
from qiskit.quantum_info import SparsePauliOp

# Method used to create QAOA circuits to optimize
from qiskit.circuit.library import QAOAAnsatz

# Tools used for the execution of quantum circuits
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Transpilation of quantum circuits on IBM's simulators and quantum computers
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Quantum computer simulator
from qiskit_aer import AerSimulator

# Simulators of existing quantum computers
from qiskit_ibm_runtime.fake_provider import FakeNairobiV2  # , FakeQuebec

# Visualization of probablity distributions
from qiskit.visualization import plot_histogram


from utils import run_circuit


class Day:
    def __init__(
        self,
        night_circuit: QuantumCircuit,
        endangered_players: list,
        roles: list,
        couple: None,
    ):
        self.night_circuit = night_circuit
        self.roles = roles
        self.endangered_players = endangered_players
        self.couple = couple

    def night_measures(self):   

        res_bitstring = run_circuit(self.night_circuit)

        killed_players = []
        for faith, player in zip(res_bitstring, self.endangered_players):
            if faith == "1":
                killed_players.append(player)
                # self.roles[player] = None

        return killed_players, self.roles

    def hunter(self, player_to_kill: int):   

        players_to_kill = [player_to_kill]     
        qc = QuantumCircuit(1)
        qc.rx(2 * np.arcsin(np.sqrt(0.9)), 0)        
        if self.couple is not None:
            if player_to_kill in self.couple:
                additional_qubit = QuantumRegister(1)
                qc.add_register(additional_qubit)
                new_qc = QuantumCircuit(qc.num_qubits)
                new_qc.append(qc, qc.qubits[:])
                new_qc.cx(0, new_qc.num_qubits-1)                
                qc = new_qc        
        res_bitstring = run_circuit(qc)        
        killed_players = []   

        for faith, player in zip(res_bitstring, players_to_kill):
            if faith == "1":
                killed_players.append(player)
            self.roles[player] = None        
                
        return killed_players, self.roles

    def vote(self, ballot: int) -> bool:
        """
        Decides if the voted player dies or not.

        Args:
            ballot (int): the position of the voted player in the players array

        returns:
            bool: True if the player has been killed, else False

        """

        circuit = QuantumCircuit(1)

        theta = np.pi * 2 / 3
        circuit.rx(theta, 0)

        result_bit = run_circuit(circuit)

        if result_bit == "1":
            self.roles[ballot] = None
            return True, self.roles

        return False, self.roles
