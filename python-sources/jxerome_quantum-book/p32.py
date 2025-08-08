#!/usr/bin/env python

import qiskit as q
import numpy as np

simulator = q.Aer.get_backend('statevector_simulator')

def main():
    b_a_ba_y()
    etat_init = etat_initial()
    resultat_h = executer(circuit_h())
    resultat_hx = executer(circuit_hx())
    resultat_hxy = executer(circuit_hxy())
    resultat_hx_simy = simulation_y(resultat_hx)

    print_vector('|psy>                   ', etat_init)
    print_vector('|psy> -> H              ', resultat_h)
    print_vector('|psy> -> H -> X         ', resultat_hx)
    print_vector('|psy> -> H -> X -> Y    ', resultat_hxy)
    print_vector('|psy> -> H -> X -> sim Y', resultat_hx_simy)

def etat_initial():
    return [np.sqrt(3)/2, (1 - 1j)/(2*np.sqrt(2))]

def nouveau_circuit():
    circuit = q.QuantumCircuit(1)
    circuit.append(q.extensions.Initialize(etat_initial()), [0])
    return circuit

def circuit_h():
    circuit = nouveau_circuit()
    circuit.h(0)
    return circuit

def circuit_hx():
    circuit = nouveau_circuit()
    circuit.h(0)
    circuit.x(0)
    return circuit

def circuit_hxy():
    circuit = nouveau_circuit()
    circuit.h(0)
    circuit.x(0)
    circuit.y(0)
    return circuit

def b_a_ba_y():
    circuit1 = q.QuantumCircuit(1)
    circuit1.append(q.extensions.Initialize([1, 0]), [0])
    circuit1.y(0)
    r0 = executer(circuit1)

    circuit2 = q.QuantumCircuit(1)
    circuit2.append(q.extensions.Initialize([0, 1]), [0])
    circuit2.y(0)
    r1 = executer(circuit2)

    print_vector('Y(|0>)', r0)
    print_vector('Y(|1>)', r1)

def simulation_y(etat):
    y_mat = np.array([[0, -1j],[1j, 0]])
    etat = np.array(etat)
    return np.matmul(y_mat, etat)


def print_vector(name: str, vector) -> None:
    print(f'{name} = [{vector[0]:.6f}; {vector[1]:.6f}]')

def executer(circuit):
    job = q.execute(circuit, simulator)
    resultat = job.result()
    return resultat.get_statevector()
    
if __name__ == '__main__':
    main()

