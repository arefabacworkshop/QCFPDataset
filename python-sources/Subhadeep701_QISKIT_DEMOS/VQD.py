from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal,NLocal,RXGate
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp,Operator
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
H = np.array([[ 0.12, 0.30, 0.00, 0.00],
 [0.3, 0.45, 0.92, 0.00],
 [ 0.00, 0.92, -0.77, 0.34],
 [ 0.00,0.00, 0.34, 0.22]])
[EIG,EV]=np.linalg.eig(H)
print(EIG)
op=Operator(H)
A=SparsePauliOp.from_operator(op)
#print(np.log2(op.dim[1]))
#b=SparsePauliOp.to_matrix(A)
#print(b)
qubits=int(np.log2(op.dim[1]))

#print(*A.paulis,A)
a=TwoLocal(qubits,["rz","ry"],"cx",entanglement="linear",reps=2)
theta=np.random.rand(a.num_parameters)
#theta = (2 * np.pi * np.random.rand(8)).tolist()
#print(theta)
qc=QuantumCircuit(qubits)
qc.x(0)
qc=qc.compose(a)
#qc.decompose().draw('mpl')



estimator=StatevectorEstimator()
sampler=StatevectorSampler()


def calculate_overlaps(ansatz, prev_circuits, parameters, sampler):
    def create_fidelity_circuit(circuit_1, circuit_2):

        """
        Constructs the list of fidelity circuits to be evaluated.
        These circuits represent the state overlap between pairs of input circuits,
        and their construction depends on the fidelity method implementations.
        """

        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

    overlaps = []

    for prev_circuit in prev_circuits:
        fidelity_circuit = create_fidelity_circuit(ansatz, prev_circuit)
        sampler_job = sampler.run([(fidelity_circuit, parameters)])
        meas_data = sampler_job.result()[0].data.meas

        counts_0 = meas_data.get_int_counts().get(0, 0)
        shots = meas_data.num_shots
        overlap = counts_0 / shots
        overlaps.append(overlap)

    return np.array(overlaps)

def cost_func_vqd(parameters, ansatz, prev_states, step, betas, estimator, sampler, hamiltonian):

    estimator_job = estimator.run([(ansatz, hamiltonian, [parameters])])

    total_cost = 0

    if step > 1:
        overlaps = calculate_overlaps(ansatz, prev_states, parameters, sampler)
        total_cost = np.sum([np.real(betas[state] * overlap) for state, overlap in enumerate(overlaps)])

    estimator_result = estimator_job.result()[0]

    value = estimator_result.data.evs[0] + total_cost

    return value
k = 3
betas = [3, 3, 3]
x0 = theta


prev_states = []
prev_opt_parameters = []
eigenvalues = []

for step in range(1, k + 1):

    if step > 1:
        prev_states.append(qc.assign_parameters(prev_opt_parameters))

    result = minimize(cost_func_vqd, x0, args=(qc, prev_states, step, betas, estimator, sampler, A),
                      method="COBYLA", options={'maxiter': 300, })
    print(result)

    prev_opt_parameters = result.x
    eigenvalues.append(result.fun)
plt.show()

