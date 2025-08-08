#use unitary simulator backend to execute the code 
qc= QuantumCircuit(1)
qc.x(0)
sv_sim = Aer.get_backend('unitary_simulator')
job= execute(qc,sv_sim)
result= job.result()
output_vector = result.get_unitary()
print(output_vector)

#make random circuit
from qiskit.circuit.random import random_circuit
qc = random_circuit(2,4)

#initialize, evolve and plot statevector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector 
qc = random_circuit(2,4)
sv = Statevector.from_label('00')

ev= sv.evolve(qc)
print(ev)
plot_state_city(ev)
ev.draw('city')

backend = BasicAer.get_backend('statevector_simulator')
job= execute(qc, backend)
result= job.result()
sv= result.get_statevector()
plot_state_city(sv)


#run and get memory
job= execute(circ,backend)
result = job.result()
counts= result.get_counts()
result = backend.run(circ, shots=10, memory=True).result()
memory = result.get_memory(circ)

# Matrix Product State simulation method
sim_mps = Aer.get_backend('aer_simulator_matrix_product_state')
job_mps = sim_mps.run(circ, shots=shots)
counts_mps = job_mps.result().get_counts(0)
plot_histogram([counts_stabilizer, counts_statevector, counts_density, 
counts_mps], title='Counts for different simulation methods', 
legend=['stabilizer', 'statevector', 'density_matrix', 'matrix_product_state'])


#convert circuit into QASM
str = qc.qasm()
print(str)
# qasm string to circuit
qc_new = QuantumCircuit.from_qasm_str(str)
qc_new.draw()

# executing circuit on real quantum computer
from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q',group='open',project='main')
provider.backends()

job = execute( qc, provider.get_backend('ibmq_lima'))
result = job.result()
counts= result.get_counts()
plot_histogram(counts)


#check state_fidelity
from qiskit.quantum_info import state_fidelity
qc1 = QuantumCircuit(2)
qc2 = QuantumCircuit(2)
qc1.h(0)
qc1.h(1)
qc2.h([0, 1])

backend= Aer.get_backend('statevector_simulator')
sv1= execute(qc1,backend).result().get_statevector(qc1)
sv2= execute(qc2,backend).result().get_statevector(qc2)

print(state_fidelity(sv1,sv2))

#operator 
from qiskit.quantum_info import average_gate_fidelity,process_fidelity
from qiskit.quantum_info.operators import Operator 
from qiskit.circuit.library import SGate
import numpy as np

gate1 = Operator(SGate())
gate2= np.exp(1j/2)*gate1
print(average_gate_fidelity(gate1,gate2))
print(process_fidelity(gate1,gate2))

# 10.3: Density Matrix:
from qiskit import quantum_info 
matrix1 = [[1+0.j, 0.5+0.j], [0.5+0.j, 1+0.j]]
matrix2 = [[0.5+0.j, 1+0.j], [0.5+0.j, 1+0.j]]
matrix = quantum_info.DensityMatrix(matrix1)
matrix.tensor(matrix2)

# 10.4 random state vector 
from qiskit import quantum_info
num_qubits = 4 
dims = 2**num_qubits 
random_state = quantum_info.random_statevector(dims = dims, seed = None) 
print(random_state)

