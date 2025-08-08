import math
from qiskit import *

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(q, c)

qc.rz(math.pi/1, q[2])
qc.cz(q[0], q[1])
qc.cx(q[2], q[1])

qc.rx(4.71238898038469,q[1])