from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import qrcode

def convert_to_binary(data):
    return ''.join(format(ord(char), '08b') for char in data)

def bernstein_vazirani_circuit(a_bin):
    n = len(a_bin)
    qc = QuantumCircuit(n + 1, n)

    # Ancilla in |1⟩
    qc.x(n)
    qc.h(n)

    # Put inputs in superposition
    qc.h(range(n))

    # Oracle: CX(i, ancilla) where a_i = 1
    for i, bit in enumerate(a_bin[::-1]):  # q0 is leftmost in Qiskit
        if bit == '1':
            qc.cx(i, n)

    # Hadamard again on input
    qc.h(range(n))

    # Measure input qubits only
    qc.measure(range(n), range(n))

    return qc

def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def main():
    data = input("Enter the text: ")

    decoded_char = ""
    length = len(data)
    for i in range(length):
        bin_str = convert_to_binary(data[i])
        print(f"Binary for '{data[i]}': {bin_str}")

        qc = bernstein_vazirani_circuit(bin_str)
        backend = AerSimulator()
        backend.set_options(shots=1024)
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=1024).result()
        counts = result.get_counts()

        print("Quantum output:", counts)
        result_bitstring = list(counts.keys())[0]
        decoded_char += chr(int(result_bitstring, 2))
        plot_histogram(counts)
        plt.show()
    print("Decoded message:", decoded_char)

    plot_histogram(counts)
    plt.show()

    qr_img = generate_qr_code(decoded_char)
    qr_img.save("qr_code.png")
    qr_img.show()

main()
