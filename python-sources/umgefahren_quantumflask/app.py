import matplotlib.axes
from flask import Flask
from flask import render_template, request
from os import environ

from time import sleep, time
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import execute, BasicAer, IBMQ, Aer
import qiskit
from qiskit.tools.visualization import plot_histogram
from qiskit.test.mock import FakeVigo

app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    return render_template("quant_field.html")

@app.route('/impressum/')
def impressum():
    variables = environ
    ImpressumName = variables["IMPRESSUM_NAME"]
    Street = variables["IMPRESSUM_ADDRESS_STREET"]
    City = variables["IMPRESSUM_ADDRESS_CITY"]
    TelefoneNumber = variables["IMPRESSUM_TELEFON"]
    EMail = variables["IMPRESSUM_EMAIL"]
    return render_template("impressum.html", ImpressumName=ImpressumName, Street=Street, City=City, TelefoneNumber=TelefoneNumber, EMail=EMail)

@app.route('/normal_computer/')
def normal_computer():
    return render_template("normal_computer.html")

@app.route('/quantum_computer/', methods=['GET', 'POST'])
def quantum_computer():
    if request.method == 'POST':
        print(request.form)
        flip = True
        if "Don't Flip" == request.form["FlipDecision"]:
            flip = False
        print(flip)
        q = QuantumRegister(1)
        c = ClassicalRegister(1)
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        if flip:
            qc.x(q[0])
        else:
            qc.id(q[0])
        qc.h(q[0])
        qc.measure(q, c)
        if request.form["BackendDecision"] == "IBMQ":
            try:
                provider = IBMQ.enable_account(environ["TOKEN"])
            except Exception as e:
                print(e)
                provider = IBMQ.load_account()
            backend_list = [
                provider.backend.ibmq_manila,
                provider.backend.ibmqx2,
                provider.backend.ibmq_belem,
                provider.backend.ibmq_16_melbourne,
                provider.backend.ibmq_lima,
                provider.backend.ibmq_armonk,
                provider.backend.ibmq_athens,
                provider.backend.ibmq_quito,
                provider.backend.ibmq_santiago
            ]
            backend = qiskit.providers.ibmq.least_busy(backend_list)
        elif request.form["BackendDecision"] == "IBMQSimulator":
            backend = FakeVigo()
        else:
            backend = Aer.get_backend('qasm_simulator')
        print(backend)
        start = time()
        job = execute(qc, backend, shots=200)

        if request.form["BackendDecision"] == "IBMQ":
            return render_template("rediriction.html", url="https://quantum-computing.ibm.com/jobs/{}".format(job.job_id()))


        result = job.result()
        print(time() - start)
        counts = result.get_counts(qc)
        qc.draw("mpl", filename="static/images/out.png")
        print(counts)
        if counts["0"] == 200:
            diagram = plot_histogram(counts)
        else:
            diagram = plot_histogram(counts)
        diagram.savefig("static/images/diagram.png")

        return render_template("quantum_computer_result.html")
    else:
        return render_template("quantum_computer.html")

if __name__ == '__main__':
    app.run()

