import math
from qiskit import *

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(q, c)

qc.y(q[0])
qc.z(q[0])
qc.rx(math.pi/7, q[1])
qc.t(q[0])
qc.h( q[0])
qc.swap(q[1], q[0])
