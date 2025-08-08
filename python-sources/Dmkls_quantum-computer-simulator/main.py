import numpy as np
import math
import fractions
from collections import Counter


class QuantumCircuit:
    def __init__(self, num_qubits: int):
        """Создаёт вектор квантовой системы и задаёт ей состояние |0...0>

        :param num_qubits: количество кубитов во всей системе
        """
        self.num_qubits = num_qubits
        self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        self.state[0] = 1

    def apply_gate(self, gate: np.ndarray, targets: list[int]):
        """Действует гейтом на заданные кубиты

        :param gate: унитарная матрица 2x2, действующая как гейт
        :param targets: массив, содержащий индексы кубитов, на которые нужно подействовать гейтом
        :return: None
        """
        full_gate = 1
        for i in range(self.num_qubits):
            if i in targets:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        self.state = full_gate @ self.state

    def apply_controlled_gate(self, gate: np.ndarray, control: int, target: int):
        """Применяет гейт к целевому кубиту при условии, что управляющий кубит находится в состоянии |1⟩.

        :param gate: унитарная матрица 2x2, представляющая гейт, который применяется к целевому кубиту при выполнении условия
        :param control: индекс управляющего кубита
        :param target: индекс целевого кубита
        :return: None
        """
        size = 2 ** self.num_qubits
        result = np.zeros((size, size), dtype=complex)
        for i in range(size):
            binary = list(format(i, f'0{self.num_qubits}b'))
            if binary[self.num_qubits - 1 - control] == '1':
                j = i ^ (1 << (self.num_qubits - 1 - target))
                temp = gate[int(binary[self.num_qubits - 1 - target])][:]
                result[i, i] = temp[0]
                result[i, j] = temp[1]
            else:
                result[i, i] = 1
        self.state = result @ self.state

    def apply_controlled_phase(self, theta: float, control: int, target: int):
        """Применяет контролируемый фазовый сдвиг к целевому кубиту, если управляющий и целевой кубиты находятся в состоянии |1⟩.

        :param theta: угол фазового сдвига в радианах
        :param control: индекс управляющего кубита
        :param target: индекс целевого кубита
        :return: None
        """
        size = 2 ** self.num_qubits
        matrix = np.eye(size, dtype=complex)
        for i in range(size):
            bin_state = format(i, f'0{self.num_qubits}b')
            if bin_state[self.num_qubits - 1 - control] == '1' and bin_state[self.num_qubits - 1 - target] == '1':
                matrix[i, i] *= np.exp(1j * theta)
        self.state = matrix @ self.state

    def measure(self, shots=1024) -> dict[str, int]:
        """Выполняет измерение квантового состояния заданное число раз и возвращает распределение результатов.

        :param shots: количество измерений (по умолчанию 1024)
        :return: словарь, содержащий строки бит в количество раз, сколько они появились
        """
        probs = np.abs(self.state) ** 2
        outcomes = [format(i, f'0{self.num_qubits}b') for i in range(2 ** self.num_qubits)]
        measurements = np.random.choice(outcomes, size=shots, p=probs)
        return dict(Counter(measurements))

# Гейты
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
P = lambda theta: np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

def qft(circuit: QuantumCircuit, qubits: list[int]):
    """Делает прямое квантовые преобразование Фурье

    :param circuit: Квантовая схема, в которой выполняется преобразование
    :type circuit: QuantumCircuit
    :param qubits: список квбитов, к которым применяется обратное QFT
    :return: None
    """
    n = len(qubits)
    for i in range(n):
        circuit.apply_gate(H, [qubits[i]])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.apply_controlled_phase(angle, qubits[j], qubits[i])

def inverse_qft(circuit: QuantumCircuit, qubits: list[int]):
    """Делает обратное квантовые преобразование Фурье

    :param circuit: Квантовая схема, в которой выполняется преобразование
    :type circuit: QuantumCircuit
    :param qubits: список квбитов, к которым применяется обратное QFT
    :return: None
    """
    n = len(qubits)
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -np.pi / (2 ** (j - i))
            circuit.apply_controlled_phase(angle, qubits[j], qubits[i])
        circuit.apply_gate(H, [qubits[i]])

def get_angles(a: int, n: int) -> np.ndarray[tuple[int], np.dtype[float]]:
    """Используется для симуляции фазовых сдвигов в функции phiADD, строит массив углов фазовых сдвигов

    :param a: Число для прибавления к квантовому регистру
    :param n: Количество кубитов в верхнем регистре
    :return: Массив из n углов, соответствующих фазовым поворотам, которые реализуют добавление числа a к квантовому регистру
    """
    s = bin(a)[2:].zfill(n)
    angles = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            if s[j] == '1':
                angles[n - i - 1] += 1 / 2 ** (j - i)
        angles[n - i - 1] *= np.pi
    return angles

def ccphase(circuit: QuantumCircuit, theta: float, ctrl1: int, ctrl2: int, target: int):
    """Выполняет дважды контролируемый фазовый сдвиг

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param theta: Угол, на котороый происходит сдвиг
    :param ctrl1: Индекс первого управляющего кубита
    :param ctrl2: Индекс второго управляющего кубита
    :param target: Целевой кубит
    :return: None
    """
    size = 2 ** circuit.num_qubits
    matrix = np.eye(size, dtype=complex)
    for i in range(size):
        bin_state = format(i, f'0{circuit.num_qubits}b')
        if bin_state[circuit.num_qubits - 1 - ctrl1] == '1' and bin_state[circuit.num_qubits - 1 - ctrl2] == '1' and bin_state[circuit.num_qubits - 1 - target] == '1':
            matrix[i, i] *= np.exp(1j * theta)
    circuit.state = matrix @ circuit.state

def phiADD(circuit: QuantumCircuit, qubits: list[int], a: int, inv=False):
    """Выполняет прибавление числа a к регистру qubits в фазовом представлении
    с помощью поразрядных фазовых поворотов

    Применяет гейты вида P(θ) к каждому кубиту регистра, имитируя сложение a в базисе
    квантового преобразования Фурье
    При `inv=True` выполняется вычитание (обратный сдвиг фазы)

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param qubits: Регистр, к которому прибавляется (или из которого вычитается) число a
    :param a: Целое число, которое прибавляется по модулю
    :param inv: Флаг, указывающий на выполнение вычитания вместо прибавления
    :return: None
    """
    n = len(qubits)
    angles = get_angles(a, n)
    for i in range(n):
        angle = -angles[i] if inv else angles[i]
        circuit.apply_gate(P(angle), [qubits[i]])


def cphiADD(circuit: QuantumCircuit, qubits: list[int], ctrl: int, a: int, inv=False):
    """Выполняет контролируемое прибавление числа a к квантовому регистру в фазовом представлении

    Реализует поразрядное прибавление фазовых сдвигов к каждому кубиту регистра,
    только если управляющий кубит `ctrl` находятся в состоянии |1⟩.
    При `inv=True` происходит вычитание (сдвиги с отрицательным углом)

    Операция применяется в базисе квантового преобразования Фурье и используется
    для построения модульной арифметики в алгоритме Шора

    :param circuit: Квантовая схема, в которой выполняется операция
    :param qubits: Регистр, к которому прибавляется (или из которого вычитается) число a
    :param ctrl: Индекс управляющего кубита
    :param a: Число, которое прибавляется (или вычитается)
    :param inv: Флаг, указывающий, нужно ли вычитать вместо прибавления (по умолчанию False)
    :return: None
    """
    n = len(qubits)
    angles = get_angles(a, n)
    for i in range(n):
        angle = -angles[i] if inv else angles[i]
        circuit.apply_controlled_phase(angle, ctrl, qubits[i])


def ccphiADD(circuit: QuantumCircuit, qubits: list[int], ctrl1: int, ctrl2: int, a: int, inv=False):
    """Выполняет дважды контролируемое прибавление числа a к регистру qubits в фазовом представлении

    Реализует поразрядное прибавление фазовых сдвигов к каждому кубиту регистра,
    только если оба управляющих кубита находятся в состоянии |1⟩
    При `inv=True` происходит вычитание (сдвиги с отрицательным углом)

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param qubits: Регистр, к которому прибавляется (или из которого вычитается) число a
    :param ctrl1: Индекс первого управляющего кубита
    :param ctrl2: Индекс второго управляющего кубита
    :param a: Число, которое прибавляется (или вычитается)
    :param inv: Флаг, указывающий, нужно ли вычитать вместо прибавления (по умолчанию False)
    :return: None
    """
    n = len(qubits)
    angles = get_angles(a, n)
    for i in range(n):
        angle = -angles[i] if inv else angles[i]
        ccphase(circuit, angle, ctrl1, ctrl2, qubits[i])


def ccphiADDmodN(circuit: QuantumCircuit, qubits: list[int], ctrl1: int, ctrl2: int, aux_bit: int, a: int, N: int):
    """Выполняет дважды контролируемое прибавление числа a по модулю N к регистру q

    Реализует следующую логику:
    - условно прибавляет a при ctrl1 = ctrl2 = 1,
    - вычитает N безусловно,
    - проверяет, произошло ли переполнение (если результат < 0),
    - в случае переполнения добавляет N обратно, чтобы результат был корректен по модулю N,
    - использует вспомогательный кубит aux_bit для хранения и последующего сброса флага переполнения


    :param circuit: Квантовая схема
    :type circuit: QuantumCircuit
    :param qubits: Регистр, к которому прибавляется число
    :param ctrl1: Индекс первого управляющего кубита
    :param ctrl2: Индекс второго управляющего кубита
    :param aux_bit: Вспомогательный кубит для отслеживания переполнения
    :param a: Число, которое прибавляется по модулю N
    :param N: Модуль
    :return: None
    """
    ccphiADD(circuit, qubits, ctrl1, ctrl2, a, inv=False)  # условное прибавление a
    phiADD(circuit, qubits, N, inv=True)  # безусловное вычитание N (получаем x + a - N)
    inverse_qft(circuit, qubits)  # проверяем переполнение
    circuit.apply_gate(X, [qubits[-1]])
    circuit.apply_controlled_gate(X, qubits[-1], aux_bit)
    circuit.apply_gate(X, [qubits[-1]])
    qft(circuit, qubits)
    cphiADD(circuit, qubits, aux_bit, N, inv=False)  # если было переполнение — прибавляем N обратно
    ccphiADD(circuit, qubits, ctrl1, ctrl2, a,
             inv=True)  # Вычитаем a обратно, чтобы восстановить начальное значение x, если переполнение произошло
    inverse_qft(circuit, qubits)  # вновь проверяем и сбрасываем aux_bit
    circuit.apply_gate(X, [qubits[-1]])
    circuit.apply_controlled_gate(X, qubits[-1], aux_bit)
    circuit.apply_gate(X, [qubits[-1]])
    qft(circuit, qubits)
    ccphiADD(circuit, qubits, ctrl1, ctrl2, a, inv=False)  # прибавляем a, так как не будет переполнения


def egcd(a: int, b: int):
    """Реализует расширенный алгоритм Евклида

    Находит наибольший общий делитель g = gcd(a, b), а также такие целые коэффициенты x и y, что:

        a * x + b * y = g

    :param a: Первое целое число
    :param b: Второе целое число
    :return: Кортеж (g, x, y), где g = gcd(a, b) и a * x + b * y = g
    :rtype: tuple[int, int, int]
    """
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y


def modinv(a: int, m: int):
    """Вычисляет обратный элемент по модулю m, то есть такое x, что a·x ≡ 1 mod m

    Использует расширенный алгоритм Евклида. Если числа не взаимно просты, вызывается исключение

    :param a: Число, для которого ищется обратный элемент
    :param m: Модуль, по которому вычисляется обратное
    :return: Обратное число x по модулю m, такое что (a * x) % m == 1
    :raises Exception: Если обратного элемента не существует (a и m не взаимно просты)
    """
    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('Обратного по модулю не существует')
    return x % m


def cMULTmodN(circuit: QuantumCircuit, ctrl: int, x_qubits: list[int], out_qubits: list[int], aux_bit: int, a: int, N: int):
    """Выполняет контролируемое умножение регистра на число a по модулю N

    Реализует операцию |x⟩ ⊗ |y⟩ → |x⟩ ⊗ |(y · a^x) mod N⟩, если управляющий кубит установлен

    Операция выполняется поразрядно в фазовом представлении с использованием квантового преобразования Фурье

    :param circuit: Квантовая схема, в которой выполняется операция
    :param ctrl: Индекс управляющего кубита, активирующего операцию
    :param x_qubits: Регистр, содержащий число x (в показателе степени)
    :param out_qubits: Регистр, который будет умножен на a^x mod N
    :param aux_bit: Вспомогательный кубит для контроля переполнения при mod N
    :param a: Множитель, основание степени
    :param N: Модуль
    :return: None
    """
    n = len(x_qubits)
    qft(circuit, out_qubits)
    for i in range(n):
        factor = (pow(2, i) * a) % N
        ccphiADDmodN(circuit, out_qubits, x_qubits[i], ctrl, aux_bit, factor, N)
    inverse_qft(circuit, out_qubits)


def initialize_shor_circuit(N: int):
    """Подготавливает начальное квантовое состояние для алгоритма Шора:

    верхний регистр инициализируется в равномерную суперпозицию,
    нижний регистр — в |1⟩, вспомогательный кубит — в |0⟩

    :param N: Число, подлежащее факторизации
    :type N: int
    :return:
        - qc (QuantumCircuit): Инициализированная квантовая схема
        - up_reg (list[int]): Верхний регистр (для значений x)
        - down_reg (list[int]): Нижний регистр (для хранения a^x mod N)
        - aux_bit (int): Вспомогательный кубит для контроля арифметики mod N
        - n (int): Количество кубитов в регистрах (разрядность)
    :rtype: tuple[QuantumCircuit, list[int], list[int], int, int]
    """

    n = math.ceil(math.log2(N))
    num_qubits = 2 * n + 1
    qc = QuantumCircuit(num_qubits)
    up_reg = list(range(0, n))
    down_reg = list(range(n, 2 * n))
    aux_bit = 2 * n
    qc.apply_gate(H, up_reg)
    qc.apply_gate(X, [down_reg[0]])

    return qc, up_reg, down_reg, aux_bit, n


def apply_controlled_exponentiation(qc, up_reg, down_reg, aux_bit, a, N):
    """Выполняет последовательное контролируемое модульное возведение в степень:

    реализует унитарное преобразование |x⟩ ⊗ |1⟩ → |x⟩ ⊗ |a^x mod N⟩.

    Для каждого бита xᵢ верхнего регистра управляемо умножает нижний регистр на a^{2^i} mod N

    :param qc: Квантовая схема, в которой происходит вычисление
    :type qc: QuantumCircuit
    :param up_reg: Верхний регистр, содержащий биты значения x
    :param down_reg: Нижний регистр, в который записывается результат a^x mod N
    :param aux_bit: Вспомогательный кубит для контроля переполнений в модульной арифметике
    :param a: Основание степени
    :param N: Модуль
    :return: None
    """
    for i in range(len(up_reg)):
        exponent = pow(a, 2 ** i, N)
        cMULTmodN(qc, up_reg[i], down_reg, down_reg, aux_bit, exponent, N)


def find_period(x: int, n: int, N: int, a: int):
    """Ищет период функции f(x) = a^x mod N на основе результата измерения квантового регистра

    :param x: Результат измерения верхнего регистра
    :param n: Количество кубитов в верхнем регистре (разрядность измерения)
    :param N: Число, подлежащее факторизации
    :param a: Основание степени, взаимно простое с N
    :return: Найденный период r, если удалось, иначе None
    :rtype: int | None
    """

    if x == 0:
        return None
    T = 2 ** n
    frac = fractions.Fraction(x, T).limit_denominator(N)
    r = frac.denominator
    if pow(a, r, N) == 1:
        return r
    return None


def measure_and_analyze(qc: QuantumCircuit, up_reg: list[int], shots=1024) -> dict[str, int]:
    """Измеряет верхний регистр схемы после применения обратного квантового преобразования Фурье

    :param qc: Квантовая схема
    :param up_reg: Список индексов кубитов верхнего регистра, подлежащих измерению
    :param shots: Количество измерений (по умолчанию 1024)
    :return: Словарь вида {'битовая строка': число повторений}
    """

    inverse_qft(qc, up_reg)
    result = qc.measure(shots=shots)
    # print("Результаты измерения:")
    # for k, v in result.items():
    #     print(f"{k} — {v} раз")
    return result


def shor(N: int, a: int, shots=1024) -> list[int] | None:
    """Выполняет полный цикл алгоритма Шора с квантовой симуляцией и классической постобработкой

    :param N: Целое число, подлежащее факторизации
    :type N: int
    :param a: Целое число, взаимно простое с N (основание степени)
    :type a: int
    :param shots: Количество симулированных измерений (по умолчанию 1024)
    :type shots: int
    :return: None. Результаты выводятся в консоль
    """

    qc, up_reg, down_reg, aux_bit, n = initialize_shor_circuit(N)
    apply_controlled_exponentiation(qc, up_reg, down_reg, aux_bit, a, N)
    result = measure_and_analyze(qc, up_reg, shots)
    for bitstring in result:
        x = int(bitstring, 2)
        r = find_period(x, 2 * n, N, a)
        if r:
            print(f"Найден период r = {r}")
            factor1 = math.gcd(pow(a, r // 2) - 1, N)
            factor2 = math.gcd(pow(a, r // 2) + 1, N)
            if factor1 not in [1, N] and factor2 not in [1, N]:
                print(f"Найдено: {factor1} × {factor2} = {N}")
                return [factor1, factor2]
    print("Не удалось найти период — попробуйте с другим 'a'")
    return None

if __name__ == "__main__":
    N = int(input("Введите N для факторизации: "))
    a = int(input(f"Введите a (взаимнопростое с {N}): "))

    if math.gcd(N, a) == 1:
        shor(N, a, shots=512)
    else:
        print(f"Пожалуйста, введите число, взаимно постое с N")
