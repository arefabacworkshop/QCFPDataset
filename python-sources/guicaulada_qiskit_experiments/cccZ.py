# Thanks to http://kth.diva-portal.org/smash/get/diva2:1214481/FULLTEXT01.pdf for helping me undestand quantum computing better.
import qiskit as qk
import math

# Controlled phase-flip -> rotates the first qubit when the second one state is 1 u1(+pi/4) -> u3(+0, +0, +pi/4)
# Controlled bit-flip -> rotates the first qubit when the second one state is 1 u3(+pi, +0, +pi)

# Triple controlled Pauli-Z gate
# I believe this function applies a phase-flip on one qubit if all other three are positive ?
# I'm not sure... If anyone knows this please let me know.
def cccZ(qc, a, b, c, d):
  qc.cu1(math.pi/4, a, d)  # u3(+0, +0, +pi/4) on a if d
  qc.cx(a, b)  # u3(+pi, +0, +pi) on a if b
  qc.cu1(-math.pi/4, b, d)  # u3(+0, +0, -pi/4) on b if d
  qc.cx(a, b)  # u3(+pi, +0, +pi) on a if b
  qc.cu1(math.pi/4, b, d)  # u3(+0, +0, +pi/4) on b if d
  qc.cx(b, c)  # u3(+pi, +0, +pi) on b if c
  qc.cu1(-math.pi/4, c, d)  # u3(+0, +0, -pi/4) on c if d
  qc.cx(a, c)  # u3(+pi, +0, +pi) on a if c
  qc.cu1(math.pi/4, c, d)  # u3(+0, +0, +pi/4) on c if d
  qc.cx(b, c)  # u3(+pi, +0, +pi) on b if c
  qc.cu1(-math.pi/4, c, d)  # u3(+0, +0, -pi/4) on c if d
  qc.cx(a, c)  # u3(+pi, +0, +pi) on a if c
  qc.cu1(math.pi/4, c, d)  # u3(+0, +0, +pi/4) on c if d

if  __name__ == '__main__':
  qr = qk.QuantumRegister(4)
  cr = qk.ClassicalRegister(4)
  qc = qk.QuantumCircuit(qr, cr)

  cccZ(qc, qr[0], qr[1], qr[2], qr[3])

  qc.measure(qr, cr)

  # loads the backend
  qasm = qk.Aer.get_backend('qasm_simulator_py')

  # simulate the circuit locally and get results
  coupling_map = [[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]]
  qasm_job = qk.execute(qc, backend=qasm, shots=2048, coupling_map=coupling_map)
  qasm_results = qasm_job.result()

  qc_counts = qasm_results.get_counts(qc)

  print(qc_counts)
