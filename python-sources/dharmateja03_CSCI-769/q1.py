from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from collections import Counter

def string_to_binary(s):
    """Convert string to binary representation"""
    binary = ''.join(format(ord(c), '08b') for c in s)
    return binary


def create_circuit(binary_string):
    """Create quantum circuit for given binary string"""
    n_qubits = len(binary_string)
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Apply X gates where there are 1s in the binary string
    for i, bit in enumerate(binary_string):
        if bit == '1':
            qc.x(i)
    
    qc.barrier()  
    qc.measure_all()
    
    return qc

def run_simulation(circuit, shots=1000):
    """Run circuit on simulator"""
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    formatted_counts = {}
    for state, count in counts.items():
        formatted_counts[state] = count
        
    return formatted_counts

def binary_to_string(binary):
    """Convert binary back to string"""
    # Reverse the binary string to correct the order
    binary = ''.join(reversed(binary))
    return ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))


test_strings = ["RIT", "QIS", "GOS"]

for test_string in test_strings:
    print(f"\nTesting string: {test_string}")
    binary = string_to_binary(test_string)
    print(f"Binary representation: {binary}")
    
    circuit = create_circuit(binary)
    print("\nCircuit:")
    print(circuit)
    
    counts = run_simulation(circuit)
    
    total_shots = sum(counts.values())
    expected_count = counts.get(binary, 0)
    accuracy = (expected_count / total_shots) * 100
    
    print(f"\nResults from {total_shots} shots:")
    for measured_binary, count in counts.items():
        measured_binary_corrected = ''.join(reversed(measured_binary))
        try:
            decoded = binary_to_string(measured_binary)
            print(f"{measured_binary_corrected} ({decoded}): {count} times")
        except:
            print(f"{measured_binary_corrected} (invalid): {count} times")
