import os
import telebot

from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram

BOT_TOKEN = os.environ.get('BOT_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я try_quantum_bot. С моей помощью можно познакомиться с миром квантовых вычислений")


# Отображение состояния на сфере Блоха
@bot.message_handler(commands=['bloch_sphere'])
def bloch_sphere_handler(message):
    text = "Набери в сообщении состояние для отображения на сфере Блоха"
    sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    bot.register_next_step_handler(sent_msg, display_bloch_sphere_handler)


def display_bloch_sphere_handler(message):
    sv = Statevector.from_label(message.text)

    result = plot_bloch_multivector(sv)

    result.savefig('bloch_state.png', dpi=300, bbox_inches="tight")
    bot.send_photo(message.chat.id, photo=open('bloch_state.png', 'rb'),
                   caption="Сфера Блоха")


# Отображение квантовой цепочки из QASM сода
@bot.message_handler(commands=['qasm_code'])
def qasm_handler(message):
    text = "Введи код на QASM, и я покажу твою цепочку"
    sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    bot.register_next_step_handler(sent_msg, qasm_code_handler)


def qasm_code_handler(message):
    qasm_code_filename = f'qasm_code_{message.chat.id}.txt'
    open(qasm_code_filename, 'w').write(message.text)

    # Build a quantum circuit
    circuit = QuantumCircuit.from_qasm_str(message.text)
    qc_fig = circuit.draw('mpl')

    qc_fig.savefig('qc.png', dpi=300, bbox_inches="tight")
    bot.send_photo(message.chat.id, photo=open('qc.png', 'rb'),
                   caption="Квантовая цепочка")

    text = "Введите /run x , чтобы запустить цепочку"
    sent_msg2 = bot.send_message(message.chat.id, text, parse_mode="Markdown")

    bot.register_next_step_handler(sent_msg2, run_qc_handler)


def run_qc_handler(message):

    shots = message.text.split()[-1]
    # Build a quantum circuit
    qasm_code_filename = f'qasm_code_{message.chat.id}.txt'
    qc = QuantumCircuit.from_qasm_str(open(qasm_code_filename, 'r').read())

    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    counts_fig = plot_histogram(counts)

    counts_fig.savefig('counts.png', dpi=300, bbox_inches="tight")
    bot.send_photo(message.chat.id, photo=open('counts.png', 'rb'), caption="Результат запуска")


@bot.message_handler(func=lambda msg: True)
def dunno_all(message):
    dunno_text = "Доступные команды: /bloch_sphere /qasm_code"
    bot.reply_to(message, dunno_text)


bot.infinity_polling()
