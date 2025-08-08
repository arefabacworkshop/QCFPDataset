import math
from qiskit import *

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(q, c)

qc.id(q[2])
qc.ccx(q[0], q[2], q[1])
qc.x(q[2])
qc.z(q[0])
qc.ry(math.pi/1, q[1])
