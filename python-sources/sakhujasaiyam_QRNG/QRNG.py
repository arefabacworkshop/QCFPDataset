import math
from qiskit import QuantumCircuit, Aer, execute

def qrng(N, x):
    random_numbers = []

    n = math.log2(N)
    n = math.ceil(n)

    for iter in range(x):
        qrng = QuantumCircuit(n)

        for i in range(n):
            qrng.h(i)

        qrng.measure_all()

        backend = Aer.get_backend("qasm_simulator")
        job = execute(qrng, backend, shots=1)
        result = job.result()
        count = result.get_counts(qrng)

        count_list = list(count.keys())
        number = 0

        for i in range(n):
            number += (int(count_list[0][-(i+1)]) * (2**i)) 

        if number > N:
            number = (2*N - number)
        
        random_numbers.append(number)

    return random_numbers

'''
@app.route('/generate_random_numbers', methods=['POST'])
def generate_random_numbers():
    max_value = int(request.json['max'])
    quantity = int(request.json['quantity'])
    
    random_numbers = qrng(max_value, quantity)
    
    return jsonify(random_numbers)


@app.route('/generate_random_numbers', methods=['POST'])
def generate_random_numbers():
    max_value = int(request.json['max'])
    quantity = int(request.json['quantity'])
    
    # Execute QRNG.py script and capture its output
    process = subprocess.Popen(['python', 'QRNG.py', str(max_value), str(quantity)], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    random_numbers = output.decode().splitlines()

    return jsonify(random_numbers)

@app.route('/')
def home():
    return app.send_static_file('HomePage.html')

if __name__ == '__main__':
    app.run(debug=True)
'''