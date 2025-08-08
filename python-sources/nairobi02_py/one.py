##saving the file
# plot_histogram(data,filename='new_hist.png')

# hist=plot_histogram(data)
# hist.savefig('new_hist.png')

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit import Aer, execute
from qiskit import qiskit, QuantumCircuit 
from qiskit.visualization import plot_histogram

qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.x(qreg_q[2])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.swap(qreg_q[2], qreg_q[3])


job=qiskit.execute(circuit,qiskit.BasicAer.get_backend('qasm_simulator'))
print(job.result().get_counts())
circuit.draw()
#h gate always makes equal superposition 

from qiskit.visualization import plot_histogram
result1=job.result().get_counts()
plot_histogram(result1)

#plot using a statevector
#plot statecity
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector 
from qiskit.visualization import plot_state_city
state=Statevector(circuit)
plot_state_city(state)

#plot bloch vector 
from qiskit.visualization import plot_bloch_vector
plot_bloch_vector(circuit,title="New Bloch Sphere")
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.quantum_info import Statevector
from qiskit.visualization import (
    plot_state_city, plot_bloch_multivector,
    plot_state_paulivec, plot_state_hinton,
    plot_state_qsphere
)

qc = QuantumCircuit(2, 2)
backend = Aer.get_backend('statevector_simulator')
result = backend.run(transpile(qc, backend)).result()
qcs = result.get_statevector(qc)
print(qc)
print(backend)
print(result)
print(qcs)
array_to_latex(qcs)
qc.draw()
qc.draw('latex')
plot_bloch_multivector(qcs)
plot_state_qsphere(qcs)
plot_state_city(qcs)
plot_state_hinton(qcs)
plot_state_paulivec(qcs)

#backend = Aer.get_backend('qasm_simulator')
#result = execute(circuit, backend, shots=1024).result()
#counts = result.get_counts()
#print(counts)


#plot histogram
from qiskit import Aer, BasicAer, execute
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.visualization import plot_histogram

q = QuantumRegister(3, 'q')
c=ClassicalRegister(3,'c')
qc = QuantumCircuit(q,c)
qc.h(q[0:3])

qasm_sim = BasicAer.get_backend('qasm_simulator')
result = execute(qc, qasm_sim).result()
counts = result.get_counts()
plot_histogram(counts)
