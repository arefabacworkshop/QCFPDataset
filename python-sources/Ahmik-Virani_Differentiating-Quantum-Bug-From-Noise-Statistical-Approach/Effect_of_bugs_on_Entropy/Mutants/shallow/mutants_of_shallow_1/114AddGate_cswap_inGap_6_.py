import math
from qiskit import *

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(q, c)

qc.cz(q[1], q[2])
qc.y(q[2])
qc.cx(q[0], q[2])
qc.cz(q[1], q[2])
qc.x(q[2])
qc.cswap(q[1], q[2], q[0])