from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from numpy import pi
num_qubits = 5

def pe_05(mqc):
    q = QuantumRegister(num_qubits)
    c = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(q, c)

    for i in range(num_qubits):
        qc.h(i)

    qc = mqc.compose(qc)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.cp(pi/512,3,4)
    qc.cp(pi/256,2,4)
    qc.cp(pi/128,1,4)
    qc.cp(pi/64,0,4)
    qc.h(0)
    qc.cp(-pi/2,0,1)
    qc.cp(-pi/4,0,2)
    qc.cp(-pi/8,0,3)
    qc.h(1)
    qc.cp(-pi/2,1,2)
    qc.cp(-pi/4,1,3)
    qc.h(2)
    qc.cp(-pi/2,2,3)
    qc.h(3)

    for i in range(num_qubits):
        qc.measure(i, i)

    backend = Aer.get_backend("aer_simulator")
    job = execute(qc, backend, shots = 10000).result().get_counts(qc)
    return job