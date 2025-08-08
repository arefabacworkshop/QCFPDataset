from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RXGate

from quantum_processing import row, column
from visualization import plot_circuit


class QAOACircuit:
    """
    Class for the QAOA circuits and relevant functions and attributes.
    :param iterations: Number of iterations of QAOA, defaults to 1
    """
    def __init__(self, weights, distance, iterations=1):
        self.iterations = iterations
        self.weights = weights
        self.distance = distance
        self.circuit = None
        self.mixing_operator_circuit = None
        n, m = weights[0].shape
        self.n = n
        self.m = m
        self.qreg_main = QuantumRegister(n * m, 'main')
        self.qreg_aux1 = QuantumRegister(m, 'auxPD2')
        self.qreg_ancilla = QuantumRegister(2, 'ancillary')
        self.creg_main = ClassicalRegister(n * m, 'Cmain')
        self.creg_aux1 = ClassicalRegister(m, 'CauxPD2')
        if distance == 'wasserstein' :
            self.qreg_aux2 = QuantumRegister(n, 'auxPD1')
            self.creg_aux2 = ClassicalRegister(n, 'CauxPD1')


    def individual_mixing_operator_main(self, circuit, edge, gate):
        i, j = edge
        circuit.cnot(self.qreg_main[i*self.m + j], self.qreg_ancilla[0])

        qubits = [self.qreg_ancilla[0]]
        if self.m > 1:
            qubits += self.qreg_main[row(i, self.m, j=j)]
            if self.n > 1:
                qubits += self.qreg_main[column(j, self.n, self.m, i=i)]
        qubits += [self.qreg_main[i*self.m + j]]

        circuit.append(gate, qubits)
        circuit.reset(self.qreg_ancilla[0])

        return circuit


    def individual_mixing_operator_aux(self, circuit, point, diagram, gate):
        if diagram == 2:
            main_qubits = column(point, self.n, self.m)
            aux_qubit = self.qreg_aux1[point]
        else:
            main_qubits = row(point, self.m)
            aux_qubit = self.qreg_aux2[point]

        circuit.x(self.qreg_ancilla)
        circuit.cnot(aux_qubit, self.qreg_ancilla[0])
        circuit.mcx(self.qreg_main[main_qubits], self.qreg_ancilla[1])
        circuit.append(gate, self.qreg_ancilla[:] + [aux_qubit])
        circuit.reset(self.qreg_ancilla)

        return circuit


    def mixing_operator(self, circuit, angle):
        """
        This function applies the mixing operator once to an existing circuit.
        """
        gate_main = RXGate(angle).control(self.n + self.m - 1)
        gate_aux = RXGate(angle).control(2)
        for i in range(self.n):
            for j in range(self.m):
                circuit = self.individual_mixing_operator_main(circuit, (i, j), gate_main)

        for j in range(self.m):
            circuit = self.individual_mixing_operator_aux(circuit, j, 2, gate_aux)

        if self.distance == 'wasserstein' :
            for i in range(self.n):
                circuit = self.individual_mixing_operator_aux(circuit, i, 1, gate_aux)

        return circuit


    def problem_operator(self, circuit, angle):
        """
        This function applies the problem operator once to an existing circuit.
        """
        for i in range(self.n):
            for j in range(self.m):
                circuit.rz(angle * self.weights[0][i, j], self.qreg_main[i*self.m + j])

        for j in range(self.m):
            circuit.rz(angle * self.weights[1][j], self.qreg_aux1[j])

        if self.distance == 'wasserstein' :
            for i in range(self.n):
                circuit.rz(angle * self.weights[2][i], self.qreg_aux2[i])

        return circuit


    def get_circuit(self):
        """
        This function builds a QAOA circuit for the dcp or Wasserstein distance
        """
        if self.circuit is None:
            params = ParameterVector('angles', length=2*self.iterations + 1)

            if self.distance == 'wasserstein' :
                circuit = QuantumCircuit(
                    self.qreg_main, self.qreg_aux1, self.qreg_aux2, self.qreg_ancilla,
                    self.creg_main, self.creg_aux1, self.creg_aux2
                )
            else:
                circuit = QuantumCircuit(
                    self.qreg_main, self.qreg_aux1, self.qreg_ancilla,
                    self.creg_main, self.creg_aux1
                )

            circuit.x(self.qreg_main)
            circuit = self.mixing_operator(circuit, params[-1])

            for layer in range(self.iterations):
                circuit = self.problem_operator(circuit, params[2*layer])
                circuit = self.mixing_operator(circuit, params[2*layer + 1])

            measure_quantum = self.qreg_main[:] + self.qreg_aux1[:]
            measure_classical = self.creg_main[:] + self.creg_aux1[:]
            if self.distance == 'wasserstein' :
                measure_quantum += self.qreg_aux2[:]
                measure_classical += self.creg_aux2[:]
            circuit.measure(measure_quantum, measure_classical)

            self.circuit = circuit

        return self.circuit


    def get_mixing_operator_circuit(self):
        """
        This function returns the mixing operator as a circuit
        """
        if self.mixing_operator_circuit is None:
            angle = Parameter("$\\beta$")
            gate_main = RXGate(angle).control(self.n + self.m - 1)
            gate_aux = RXGate(angle).control(2)

            if self.distance == 'wasserstein' :
                circuit = QuantumCircuit(
                    self.qreg_main, self.qreg_aux1, self.qreg_aux2, self.qreg_ancilla
                )
            else:
                circuit = QuantumCircuit(
                    self.qreg_main, self.qreg_aux1, self.qreg_ancilla
                )

            for i in range(self.n):
                for j in range(self.m):
                    circuit = self.individual_mixing_operator_main(circuit, (i, j), gate_main)
                    circuit.barrier()

            for j in range(self.m):
                circuit = self.individual_mixing_operator_aux(circuit, j, 2, gate_aux)
                if self.distance == 'wasserstein' or j < self.m - 1 :
                    circuit.barrier()

            if self.distance == 'wasserstein' :
                for i in range(self.n):
                    circuit = self.individual_mixing_operator_aux(circuit, i, 1, gate_aux)
                    if i < self.n - 1 :
                        circuit.barrier()

            self.mixing_operator_circuit = circuit

        return self.mixing_operator_circuit


    def plot_mixing_operator(self, name):
        figure_name = 'mixing-operator-' + self.distance + '-' + name
        plot_circuit(self.get_mixing_operator_circuit(), figure_name=figure_name)
