
import socket
import pickle
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def e91_protocol_and_receive():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 65432))  # Listen on this port
        s.listen()
        print("Waiting for connection from Alice...")
        conn, addr = s.accept()
        with conn:
            data = pickle.loads(conn.recv(4096))
            alice_settings, alice_results, bob_settings, bob_results = data
            print(f"Bob received: {alice_settings}, {alice_results}")

            # Bob measures the entangled pair
            bob_measured = []
            for i in range(len(alice_settings)):
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)  # Entangled pair

                if bob_settings[i] == 0:
                    qc.measure(1, 1)
                else:
                    qc.h(1)
                    qc.measure(1, 1)

                # Simulate the quantum circuit
                simulator = Aer.get_backend('qasm_simulator')
                result = execute(qc, simulator, shots=1).result()
                counts = result.get_counts()

                # Bob's measurement results
                bob_measured.append(int(counts.get('00', 0) == 1))

            # Send Bob's results back to Alice
            conn.sendall(pickle.dumps(bob_measured))
            return bob_measured

# Run Bob's part: receive and measure
bob_results = e91_protocol_and_receive()
print(f"Bob's results: {bob_results}")
