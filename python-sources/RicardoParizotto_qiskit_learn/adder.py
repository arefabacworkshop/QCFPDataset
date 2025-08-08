

#https://learn.qiskit.org/course/introduction/the-atoms-of-computation#the-15-38

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

qc = QuantumCircuit(4,2)    #quantum circuit o 3 qbits

qc.x(0)
qc.x(1)

qc.cx(0,1)
qc.cx(0,2)   # CNOT / XOR


qc.ccx(0,1,3)  #toffoli / AND

qc.measure(2,0)
qc.measure(3,1)

qc.draw()

sim = AerSimulator()  #instantiate simulator

job = sim.run(qc)   #run the quantum circuit
result = job.result()   

result.get_counts()
qc.draw()






