'''
this is a module for a concept test for the attack to the rsa system
'''

from tinyrsa import RSA,TextEncrypter,TextDecrypter
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from fractions import Fraction
from math import gcd,log2
from quantum import constant_modular_exponentiation,eval_circuit,get_results
from random import choice
from tools import printtiming
from sympy import primerange

def x21mod143():
    '''
    return the circuit to compute a*21 mod 143
    '''
    qc = QuantumCircuit(8)
    qc.ccx(0,1,5)
    qc.ccx(0,1,3)
    qc.cx(0,5)
    qc.ccx(0,5,1)
    qc.ccx(0,5,3)
    qc.ccx(0,5,4)
    qc.mcx([0,1,5],3)
    qc.mcx([0,1,5],4)
    qc.cx(0,3)
    qc.mcx([0,1,5],6)
    qc.mcx([0,1,5],7)
    qc.cx(0,5)
    qc.mcx([0,4,5],6)
    qc.mcx([0,4,5],7)
    qc.mcx([0,1,3],4)
    qc.mcx([0,1,3],5)
    qc.cx(0,3)
    qc.name = 'controlled x 21 mod 143'
    return qc

def crack():
    '''
    return the circuit to factorize the number 143
    '''
    qc = QuantumCircuit(15)
    x21mod143_circuit = x21mod143()
    for i in range(8):
        qc.h(i)
        c = x21mod143_circuit.power(2**i)
        c.name = f'x 21^{2**i} mod 143'
        qc.append(c,[i] + list(range(8,15)))
        pass
    qc.append(QFT(8),range(8))
    return qc

def isperfectpower(n:int):
    exp_opts = primerange(1,int(log2(n)) + 1)
    for opt in exp_opts:
        a = round(n**(1/opt))
        if a**opt == n:
            return a
        pass
    return None

def check_period(period:int,func=lambda x: 21**x % 143):
    x = 1
    while True:
        if func(x) == func(x + period) and func(x) == 1:
            return True
        if func(x) != func(x + period):
            return False
        x += 1
        pass
    pass

def get_factors(val:int,base:int = 21,N:int=143):
    factor = 1
    offset = -3
    option = Fraction(val,N).denominator
    while not check_period(option*factor,lambda x: base**x % N) and factor < 4:
       factor += 1
       pass
    if factor < 4 and option * factor % 2 == 0:
        return gcd((base ** (option * factor >> 1) - 1),N) , gcd((base ** (option * factor >> 1) + 1),N)
    while offset < 4:
        factor = 1
        while not check_period((option + offset)*factor,lambda x: base**x % N) and factor < 4:
            factor += 1
            pass
        if factor < 4 and ((option + offset) * factor) % 2 == 0:
            return gcd((base ** ((option + offset) * factor >> 1) - 1),N) , gcd((base ** ((option + offset) * factor >> 1) + 1),N)
        offset += 1
        pass
    return None
    
@printtiming
def Shor(N:int):
    if N % 2 == 0:
        return N // 2, 2
    base = isperfectpower(N)
    if base:
        return base,N // base
    options = [i for i in range(2,N)]
    factors = None
    while True:
        if len(options) == 0:
            return None
        base = choice(options)
        options.remove(base)
        if gcd(base,N) == 1:
            print(f'buildig quantum circuit for {base}^x mod {N} function')
            circuit = constant_modular_exponentiation(base,N,N.bit_length())
            qc = QuantumCircuit(circuit.num_qubits,N.bit_length())
            for i in range(N.bit_length()):
                qc.h(i)
                pass
            qc = qc.compose(circuit,range(circuit.num_qubits))
            qc.append(QFT(N.bit_length()),range(N.bit_length()))
            qc.measure(range(N.bit_length()),range(N.bit_length()))
            print(f'executing circuit')
            results = get_results(**eval_circuit(qc,shots=100))
            for _,val,_ in results:
                if val == 0: continue
                factors = get_factors(val,base,N)
                if factors != None and factors[0] != 1 and factors[1] != 1:
                    return factors
                pass
            pass
        else:
            d = gcd(base,N)
            return d, N // d
        pass
    pass
