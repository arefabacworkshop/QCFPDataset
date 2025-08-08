# colour_code_executor_style.py

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization.timeline.types import DataTypes
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

g_simulator = AerSimulator() #a global simulator (why re-initiate each time?)




def generate_colour_code_lut():
    NUM_QUBITS = 7
    NUM_STABILIZERS = 6


    x_stabilizers = {
        0: [0],
        1: [0, 2],
        2: [0, 1],
        3: [0, 1, 2],
        4: [1],
        5: [1, 2],
        6: [2],
    }

    z_stabilizers = {
        0: [3],
        1: [3, 5],
        2: [3, 4],
        3: [3, 4, 5],
        4: [4],
        5: [4, 5],
        6: [5],
    }

    def syndrome_for_error(qubit, pauli):
        syndrome = [0] * NUM_STABILIZERS
        if pauli == 'X':
            for s in z_stabilizers.get(qubit, []):
                syndrome[s] = 1
        elif pauli == 'Z':
            for s in x_stabilizers.get(qubit, []):
                syndrome[s] = 1
        elif pauli == 'Y':
            for s in z_stabilizers.get(qubit, []):
                syndrome[s] = 1
            for s in x_stabilizers.get(qubit, []):
                syndrome[s] = 1
        return syndrome

    lut = {}

    for q in range(NUM_QUBITS):
        for pauli in ['X', 'Z', 'Y']:
            syn = syndrome_for_error(q, pauli)
            syndrome_str = int(''.join(str(b) for b in syn),2)

            if syndrome_str not in lut:
                lut[syndrome_str] = [(q, pauli)]

    return lut



def apply_correction(logical, correction_ops):
    corrected = logical.copy()
    for qubit, pauli in correction_ops:
        if pauli in ['X', 'Y']:
            corrected[qubit] ^= 1
    return corrected

LUT = generate_colour_code_lut()

x_stabilizer_matrix = [
    np.array([1, 1, 1, 1, 0, 0, 0]),  # X1 X2 X3 X4 → 0 1 2 3
    np.array([0, 0, 1, 1, 1, 1, 0]),  # X3 X4 X5 X6 → 2 3 4 5
    np.array([0, 1, 0, 1, 0, 1, 1]),  # X2 X4 X6 X7 → 1 3 5 6
]


from itertools import combinations

def is_degenerate(error_vector, stabilizer_matrix=x_stabilizer_matrix):
    error_vector = np.array(error_vector)
    if np.all(error_vector == 0):
        return False
    for r in range(len(stabilizer_matrix) + 1):
        for combo in combinations(stabilizer_matrix, r):
            total = np.zeros(len(error_vector), dtype=int)
            for vec in combo:
                total = (total + vec) % 2
            if np.array_equal(total, error_vector):
                return True
    return False




def depolarizing_error(qc, p, qubits):
    probs, events = [1-p, p/3, p/3, p/3], ["I", "X", "Y", "Z"]
    for q in qubits:
        err = random.choices(events, probs)[0]
        if   err == "X": qc.x(q)
        elif err == "Y": qc.y(q)
        elif err == "Z": qc.z(q)
    return qc


def colour_code(p=None):
    ar_x = QuantumRegister(3,  "ar_x")
    ar_z = QuantumRegister(3,  "ar_z")
    cl_x = ClassicalRegister(3, "cl_x")
    cl_z = ClassicalRegister(3, "cl_z")
    data = QuantumRegister(7,  "data")
    cl_d = ClassicalRegister(7, "cl_data")
    qc = QuantumCircuit(ar_x, ar_z, cl_x, cl_z, data, cl_d)

    # X-prep
    x_stabs = [(0,[0,1,2,3]), (1,[2,3,4,5]), (2,[1,3,5,6])]
    for anc, qubits in x_stabs:
        qc.h(ar_x[anc])
        for q in qubits: qc.cx(ar_x[anc], data[q])
        qc.h(ar_x[anc])
    qc.barrier()

    # depolarising noise (for demo, we just flip data[6])
    depolarizing_error(qc, p, [data[i] for i in range(7)])
    qc.barrier()

    # syndrome extraction
    for anc, qubits in x_stabs:
        qc.h(ar_x[anc])
        for q in qubits: qc.cx(ar_x[anc], data[q])
        qc.h(ar_x[anc])
        qc.measure(ar_x[anc], cl_x[anc])

    z_stabs = [(0,[0,1,2,3]), (1,[2,3,4,5]), (2,[1,3,5,6])]
    for anc, qubits in z_stabs:
        for q in qubits: qc.cx(data[q], ar_z[anc])
        qc.measure(ar_z[anc], cl_z[anc])

    qc.barrier()
    qc.measure(data, cl_d)
    return qc



def simulate_circuit(qc,LUT):
    LUT = LUT
    simulator = g_simulator
    res = simulator.run(qc, shots = 1).result()
    counts = res.get_counts()

    logical = list(counts.keys())[0][:7][::-1]
    logical = [int(i) for i in logical]

    z_stab = list(counts.keys())[0][8:11][::-1]
    x_stab = list(counts.keys())[0][12:16][::-1]

    for_lut = x_stab + z_stab
    for_lut = int(for_lut, 2)

    if for_lut in LUT:
        res = apply_correction(logical, LUT[for_lut])
    else:
        res = logical


    #
    # if is_degenerate(np.array(res)):
    #     return (True,0)

    if_error = any(res)

    return (False, if_error)


def run_trials_for_p(p,n, LUT):
    # each argument is a tuple (p, number of shots)
    total_errors = 0
    degen_count = 0
    for _ in range(n):
        is_degen, error = simulate_circuit(qc=colour_code(p), LUT=LUT)
        total_errors += error
        if is_degen:
            degen_count +=1
    avg_error = total_errors / n
    degen_ratio = degen_count / n
    return (p, avg_error, degen_ratio)

if __name__ == "__main__":
    """
    Feel free to change these just make sure you change p_values across the other files as well (if your plotting to compare that is)
    Change n if your computer is pretty slow, it may take a while to run.
    """
    # --- Parameters ---
    n = 2500  # number of trials per p
    p_values = np.arange(0.0001, 0.2, 0.005)
   # p_values = np.linspace(0.001, 0.5, 40)


    LUT = generate_colour_code_lut()
    results = []
    with ProcessPoolExecutor(max_workers = 13) as executor:
        futures = [executor.submit(run_trials_for_p, p, n, LUT) for p in p_values]
        for future in futures:
            results.append(future.result())

    import numpy as np
    results.sort()  # sort by p-value
    ps, qbers, degen_ratio = zip(*results)

    """
    save files - make sure qbers & degen_ratios are saved
    """
    np.save("color.npy", qbers)
    #np.save("color_nondegen_comp.npy", np.array(qbers))
    #np.save("degen_ratios_color.npy", np.array(degen_ratio))

    import matplotlib.pyplot as plt

    for p, qber, degen in results:
        print(f"p = {p:.5f} → QBER = {qber:.5f}, degen ratio: {degen:.5f}")

    plt.plot(ps, qbers, marker='o', ms=3)
    plt.xlabel('Depolarizing Probability (p)')
    plt.ylabel('QBER')
    plt.yscale("log")
    plt.grid(True, which= "both")
    plt.title('Colour Code: QBER vs. Depolarizing Probability')
    plt.show()

