from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class QuantumRandomNumber():
    def __init__(self, random_number, qc):
        self.random_number = random_number
        self.qc = qc
        pass

    def __str__(self):
        return f'Quantum circuit:\n{self.qc.draw('text')}\nGenerated the random number: {self.random_number}'

def qrng(number):
    '''
    - This function generates a random number by simulating the execution of a quantum circuit. 
    - The parameter "number" sets the upper limit of the number generated.
    - This function generates the random number using AerSimulator, a high performance simulator for quantum circuits that includes realistic noise models.
    - Per the Qiskit documentation: "Note that this local simulator is only possible for a small circuit. When you scale up, you will need to use a real device".
    '''
    while True:
        # Convert to binary and check how many bits in contains (which will be the number of qubits required in the ciruit)
        number_binary = bin(number)
        num_of_bits = len(number_binary[2:])
        
        # Create a circuit with as many qubits as bits in the binary number
        qc = QuantumCircuit(num_of_bits)

        # Set a Hadamard gate in each qubit
        for i in range(0, num_of_bits):
            qc.h(i)

        qc.measure_all()

        qc_transpiled = transpile(qc) # Use 'transpile' to optimize the circuit
        
        # Execute the circuit with AerSimulator
        sim = AerSimulator()
        result = sim.run(qc_transpiled, shots=1).result()
        random_number_binary = list(result.get_counts())[0]

        # Convert random number to decimal
        random_number = int(random_number_binary, 2)

        # Check if the generated number is greater than the desired upper threshold.
        # This is done because a quantum circuit with 'n' qubits can output a number with all 1s in each position, which might be a larger number than the desired upper threshold.
        # If that is the case, re-run the qrng until we obtain a result within the desired threshold.
        if random_number <= number:
            random_number_instance = QuantumRandomNumber(random_number, qc)
            break
        
    return random_number_instance