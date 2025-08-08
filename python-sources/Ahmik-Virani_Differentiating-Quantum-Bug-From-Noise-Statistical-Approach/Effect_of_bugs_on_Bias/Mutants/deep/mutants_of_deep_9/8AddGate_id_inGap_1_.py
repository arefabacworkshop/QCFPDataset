import math
from qiskit import *

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(q, c)

qc.id( q[2])
qc.rzz(math.pi/5, q[1], q[2])
qc.y(q[2])
qc.cz(q[2], q[1])
