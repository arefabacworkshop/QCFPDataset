import time
import random
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from mido import Message, open_output, get_output_names
from PIL import Image, ImageDraw, ImageFont

# ---------- CONFIGURACIÓN ----------
PORT_NAME = 'loopMIDI Port 3'
BPM = 80
BEAT_DURATION = 60 / BPM
NOTE_DURATION = int((BEAT_DURATION / 3) * 1000)  # ms base
BASE_NOTE = 60  # C4
NUM_QUBITS = 9
SHOTS = 2
USE_LFO = True
# -----------------------------------

# Inicializar simulador
simulator = Aer.get_backend('qasm_simulator')

# Abrir puerto MIDI
print("Puertos MIDI disponibles:")
for name in get_output_names():
    print(f" - {name}")

midi_out = open_output(PORT_NAME)
print(f"\n🎵 Secuenciador cuántico extendido en '{PORT_NAME}'")
print("Presiona Ctrl+C para detener...\n")

# LFO cuántico para velocity
def get_quantum_lfo(num_bits=5, mod_range=60, offset=50):
    lfo_qc = QuantumCircuit(num_bits, num_bits)
    for i in range(num_bits):
        lfo_qc.h(i)
    lfo_qc.measure(range(num_bits), range(num_bits))
    
    result = simulator.run(lfo_qc, shots=1).result()
    bitstring = list(result.get_counts())[0].replace(" ", "")
    value = int(bitstring, 2)
    
    max_val = 2 ** num_bits - 1
    scaled = int((value / max_val) * mod_range)
    
    return offset + scaled

class QuantumSmoothLFO:
    def __init__(self, min_val=20, max_val=200, step_size=10):
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = step_size
        self.state = (min_val + max_val) // 2  # Valor inicial
        self.simulator = Aer.get_backend('qasm_simulator')

    def step(self):
        # Circuito simple de un solo qubit
        qc = QuantumCircuit(1, 1)
        qc.h(0)  # Estado superpuesto
        qc.measure(0, 0)
        result = self.simulator.run(qc, shots=1).result()
        counts = result.get_counts(qc)
        bit = list(counts.keys())[0].replace(" ", "")

        # Si es '1' sube, si es '0' baja
        if bit == '1':
            self.state = min(self.state + self.step_size, self.max_val)
        else:
            self.state = max(self.state - self.step_size, self.min_val)

        return self.state

# Crear circuito con puertas aleatorias
def create_quantum_circuit():
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    for i in range(NUM_QUBITS):
        gate = random.choice(["h", "x", "rz"])
        if gate == "h":
            qc.h(i)
        elif gate == "x":
            qc.x(i)
        elif gate == "rz":
            qc.rz(random.uniform(0.1, 3.14), i)
    
    # Swap aleatorio entre pares
    if NUM_QUBITS >= 2 and random.random() < 0.4:
        a, b = random.sample(range(NUM_QUBITS), 2)
        qc.swap(a, b)
    
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 7)
    qc.measure_all()
    return qc

try:

    lfo_duration = QuantumSmoothLFO(min_val=50, max_val=250, step_size=random.randint(15, 25))
    q1c = create_quantum_circuit()
    print(q1c)
 
# Obtener el diagrama ASCII
    ascii_diagram = q1c.draw(output='text').__str__()

    explicacion = (

"\n"
"\n"
"\n"
"# Crear circuito cuántico:\n"
"# 8 qubits y 8 bits clásicos\n"
"  qc = QuantumCircuit(8, 8)\n"
"\n"
"# Apply Hadamard gate to qubit 0\n"
"  qc.h(0)\n"
"\n"
"# Apply CNOT gate between qubit 0 and 1\n"
"  qc.cx(0, 1)\n"
"\n"
"# Measure qubits into classical bits\n"
"  qc.measure_all()\n"

)

    ascii_diagram = ascii_diagram + explicacion

# ────────────────────────────────
# Generar imagen tipo consola (opcional)
# ────────────────────────────────

# Configuración de imagen
    lines = ascii_diagram.split('\n')
    font_size = 18

# Usa una fuente monoespaciada (ajustá la ruta si es necesario)
    try:
     font = ImageFont.truetype("CascadiaMono.ttf", font_size)
    except:
     font = ImageFont.load_default()

# Tamaño aproximado
    max_width = max(len(line) for line in lines)
    img_width = max_width * (font_size // 2 + 3)
    img_height = len(lines) * (font_size + 4)

# Crear imagen fondo oscuro
    img = Image.new("RGB", (img_width, img_height), color=(0, 51, 102))
    draw = ImageDraw.Draw(img)

# Dibujar líneas de texto
    for i, line in enumerate(lines):
     draw.text((10, i * (font_size + 4)), line, font=font, fill=(173, 216, 230))

# Guardar imagen
    img.save("ascii_quantum_circuit_rtmidi.png")

    while True:

        qc = create_quantum_circuit()
        job = simulator.run(qc, shots=SHOTS)
        result = job.result()
        counts = result.get_counts(qc)

        for bitstring, repetitions in counts.items():
            note_value = int(bitstring.replace(" ", ""), 2) % 36
            midi_note = BASE_NOTE + note_value

            for _ in range(repetitions):
                velocity = get_quantum_lfo() if USE_LFO else random.randint(60, 100)
                midi_out.send(Message('note_on', note=midi_note, velocity=velocity))
                
                dur = lfo_duration.step() / 1000
                time.sleep(dur)

                #time.sleep(BEAT_DURATION / 3)
                midi_out.send(Message('note_off', note=midi_note, velocity=velocity))

                if random.randint(1, 100) >= 50:
                     bend = get_quantum_lfo(num_bits=5, mod_range=500, offset=0)
                     midi_out.send(Message('pitchwheel', pitch=bend - 4096))


        time.sleep(BEAT_DURATION)

except KeyboardInterrupt:
    print("\n🛑 Secuenciador detenido.")
    midi_out.close()
