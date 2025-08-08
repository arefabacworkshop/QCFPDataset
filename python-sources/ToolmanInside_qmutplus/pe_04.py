from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from numpy import pi
num_qubits = 4

def pe_04(mqc):
    q = QuantumRegister(num_qubits)
    c = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(q, c)

    for i in range(num_qubits-1):
        qc.h(i)

    qc = mqc.compose(qc)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.cp(pi/512,2,3)
    qc.cp(pi/256,1,3)
    qc.cp(pi/128,0,3)
    qc.h(0)
    qc.cp(-pi/2,0,1)
    qc.cp(-pi/4,0,2)
    qc.h(1)
    qc.cp(-pi/2,1,2)
    qc.h(2)

    for i in range(num_qubits):
        qc.measure(i, i)

    backend = Aer.get_backend("aer_simulator")
    job = execute(qc, backend, shots = 10000).result().get_counts(qc)
    return job