
import socket
import pickle
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def eavesdrop():
    # Eve intercepts the message
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65432))  # Alice's IP and port
        s.sendall(b'Eve is listening...')
        
        # Eve intercepts and tries to measure the qubits
        data = pickle.loads(s.recv(4096))
        alice_settings, alice_results, bob_settings, bob_results = data
        
        # Eve performs measurement on the qubits
        eve_results = []
        for i in range(len(alice_settings)):
            qc = QuantumCircuit(1, 1)
            if alice_settings[i] == 0:
                qc.measure(0, 0)  # Z-basis
            else:
                qc.h(0)  # X-basis
                qc.measure(0, 0)

            # Simulate the quantum circuit (Eve's measurement)
            simulator = Aer.get_backend('qasm_simulator')
            result = execute(qc, simulator, shots=1).result()
            eve_results.append(int(result.get_counts().most_frequent() == '0'))

        # Eve tries to send her results back to Alice (but this could cause mismatches)
        s.sendall(pickle.dumps(eve_results))
        return eve_results

# Run Eve's part - eavesdrop
eve_results = eavesdrop()
print(f"Eve's results: {eve_results}")
