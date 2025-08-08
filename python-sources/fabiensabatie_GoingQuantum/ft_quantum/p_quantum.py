import getpass, time, sys
from colors import bcolors as color
from random import shuffle
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit,  available_backends, execute, register, get_backend
from qiskit.tools.visualization import plot_histogram
from qiskit import register, available_backends, get_backend

def init():
	print('')
	sim = 0
	for i in range(len(sys.argv)):
		if (sys.argv[i] == '-s'):
			print('The circuit will be running on a simulator.')
			sim = 1
	# Get the user's IBM Q Experience API token
	if (sim == 0):
		print("Now attempting to get you connected to the IBM Q Experience :", end='')
		APItoken = input("Please enter your API token : ")
		qx_config = {
			"APItoken": APItoken,
			"url":"https://quantumexperience.ng.bluemix.net/api"
		}
		try:
			register(qx_config['APItoken'], qx_config['url'])
			print(color.OKGREEN, 'You have access to great power !', color.ENDC)
		except:
			print(color.FAIL, 'Something went wrong, is your token correct?', color.ENDC)
	return (sim)

def present_backends(sim, backends):
	if (sim == 0):
		# Gathering backend information
		print('Please select a backend: ')
		backends.sort()
		#Listing computers
		print('\n\033[92m###### Quantum Computers: ######\033[0m')
		print('IBMQX5 has 16 QBits - IBMQX4 and IBMQX2 have 5 QBits')
		for idx, backend in enumerate(backends):
			if backend.find('ibmqx') == 0:
				print(backend,'(',idx,')')

		#Listing simulators
		print('\n\033[92m########## Simulators ##########\033[0m')
		for idx, backend in enumerate(backends):
			if backend.find('ibmqx'):
				print(backend,'(',idx,')')
		res = input()
		return (get_backend(backends[int(res)]))
	else:
		return (get_backend('local_qasm_simulator'))

def go_job(backend, qc, b_print, shots):
	if (b_print):
		print(color.OKGREEN, 'Backend ', backend.name, ' is available : ', backend.status, color.ENDC, sep='')
		print('Submitting the job to backend', backend.name, '\n')
	job = execute(qc, backend, shots=shots)
	return (job.result().get_counts())

def getBitString(sim):

	qbits = 9
	qr = QuantumRegister(qbits)
	cr = ClassicalRegister(qbits)
	qc = QuantumCircuit(qr, cr)

	for i in range(qbits):
		qc.h(qr[i])

	for i in range(qbits):
		qc.measure(qr[i], cr[i])

	backends = available_backends();
	backend = present_backends(sim, backends)
	if backend.status["available"] is True:
		stats = go_job(backend, qc, False, 10000)
		max = 0
		for bitString in stats:
			if (stats[bitString] > max):
				max = stats[bitString]
		for bitString in stats:
			if (stats[bitString] == max):
				return (bitString)
	else:
		print(color.FAIL, '  Backend', backend.name, 'is unavailable, try again later.', color.ENDC)

def print_map(res, max):
	print('The', len(res), 'following results got measured', max, 'times')
	for i, bitString in enumerate(res): #Add checkwin
		print('___________')
		for idx, bit in enumerate(bitString[::-1]):
			if (bit == "0"):
				print('o', end='    ')
			else:
				print('x', end='    ')
			if ((idx + 1) % 3 == 0):
				print("\n")
