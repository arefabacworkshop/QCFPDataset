from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate

class AdderNode:
    def __init__(self, num_qubits, a_init=None, b_init=None):
        """
        Create a quantum adder circuit that adds two numbers encoded in quantum registers:
        A + B = C, with carry and an all-ones flag (quasi carry).

        Qubit layout:
            A:  the first quantum number of length num_qubits 
            B:  the first quantum number of length num_qubits
            C:  A+B of length num_qubits
            q (all-ones):    the quasi carry flag
            r:  the carry flag

        The final state:
        sum_qubits[0..num_qubits-1] = A+B (sum bits)
        sum_qubits[num_qubits] = final carry
        all_ones_qubit = 1 if all sum bits are 1, else 0
        """

        # Total qubits = A(n) + B(n) + Sum(n+1) + AllOnes(1) = 3*n + 2
        self.num_qubits = num_qubits
        self.total_qubits = 3 * num_qubits + 2
        self.a_qubits = QuantumRegister(num_qubits, 'a')
        self.b_qubits = QuantumRegister(num_qubits, 'b')
        self.sum_qubits = QuantumRegister(num_qubits, 'c')
        self.carry_qubit = QuantumRegister(1, 'r')
        self.qcarry_qubit = QuantumRegister(1, 'q')
        
        self.qc = QuantumCircuit(self.a_qubits, self.b_qubits, self.sum_qubits, self.carry_qubit, self.qcarry_qubit)
        if a_init != None:
            self.qc.initialize(a_init, self.a_qubits)
        
        if b_init != None:
            self.qc.initialize(b_init, self.b_qubits)

    def compute_sum(self):
        
        # Implement the ripple-carry addition
        for i in range(self.num_qubits):
            if(i==0):
                self.qc.ccx(self.a_qubits[i], self.b_qubits[i], self.carry_qubit)
            else:
                self.qc.ccx(self.a_qubits[i], self.b_qubits[i], self.carry_qubit)
                self.qc.ccx(self.a_qubits[i], self.sum_qubits[i], self.carry_qubit)
                self.qc.ccx(self.sum_qubits[i], self.b_qubits[i], self.carry_qubit)
            self.qc.cx(self.a_qubits[i], self.sum_qubits[i])
            self.qc.cx(self.b_qubits[i], self.sum_qubits[i])

            if(i+1<self.num_qubits):
                self.qc.swap(self.carry_qubit, self.sum_qubits[i+1])

        # Set the all_ones_qubit if all sum bits are 1 using MCXGate
        controls = list(self.sum_qubits)
        self.qc.append(MCXGate(num_ctrl_qubits=self.num_qubits), controls + [self.qcarry_qubit])

    def get_circuit(self):
        """Retrieves the circuit for the adder module. The order of the registers are as follows:

        Returns:
           a_qubits, self.b_qubits, self.sum_qubits, self.carry_qubit, self.qcarry_qubit
        """
        return self.qc.to_instruction(label='Adder')
    

if __name__ == "__main__":
    # Number of qubits required per input
    num_qubits = 2
    start_qubit = 0  # Start index for the qubits in the circuit

    # Create the quantum adder circuit
    adder_circuit = AdderNode(num_qubits)
   
    # print(adder_circuit.draw("text"))
