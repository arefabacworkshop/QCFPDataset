import numpy as np

from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute
from qiskit.tools.jupyter import *

from qiskit.visualization import *

from ibm_quantum_widgets import *

from qiskit.providers.aer import QasmSimulator

provider = IBMQ.load_account()

qc1 = QuantumCircuit(2,2)
qc1.h(1)

qc1.cx(0,1)

qc1.cx(1,0)

qc1.draw()
from qiskit.quantum_info import Statevector
state = Statevector.from_int(1,4)
state = state.evolve(qc1)
state.draw('latex')
