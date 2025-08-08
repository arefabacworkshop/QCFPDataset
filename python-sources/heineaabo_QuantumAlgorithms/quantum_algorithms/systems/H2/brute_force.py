import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

import sys
sys.path.append('../..')
from vqe import VQE
from optimizers import QuantumGradientDescent
sys.path.append('../../../../QuantumCircuitOptimizer')
from quantum_circuit import QuantumCircuit,SecondQuantizedHamiltonian


from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4


def get_H2_info(R,
                       basis='sto-3g',
                       multiplicity=1,
                       charge=0):
    geometry = [['H',[0,0,0]],
                ['H',[0,0,R]]]
    h2_molecule = MolecularData(geometry,basis,multiplicity,charge)

    h2_molecule = run_psi4(h2_molecule,
                            run_mp2=True,
                            run_cisd=True,
                            run_ccsd=True,
                            run_fci=True)
    one_body = h2_molecule.one_body_integrals
    two_body = h2_molecule.two_body_integrals
    Enuc = h2_molecule.nuclear_repulsion
    
    energies = {}
    energies['fci'] = h2_molecule.fci_energy
    energies['hf'] = h2_molecule.hf_energy
    energies['cisd'] = h2_molecule.cisd_energy
    energies['ccsd'] = h2_molecule.ccsd_energy
    #print('electron:',h2_molecule.n_electrons)

    return one_body, two_body, Enuc, energies

l = 4
n = 2

R = 0.7

print('For bond length {}'.format(R))
# Get psi4 
h_pq,h_pqrs,Enuc,energies = get_H2_info(R)
# Save energies
cisd = energies['cisd']
ccsd = energies['ccsd']
fci = energies['fci']
hf = energies['hf']

# Prepare circuit list
H2 = SecondQuantizedHamiltonian(n,l,h_pq,h_pqrs,nuclear_repulsion=Enuc,add_spin=True)

grid_points = 60
x = np.linspace(-0.1,0.05,grid_points)
#params = [[i] for i in x]
params = []
for i in x:
    for j in x:
        params.append(np.array([i,j]))
Es = np.zeros(grid_points*grid_points)

shots=1000
options = {'shots':shots}
i = 0
for theta in tqdm(params):
    vqe = VQE(H2,
            QuantumGradientDescent('RMSprop'),
            ansatz = 'RY',
            options=options)
    #print('{} / {}'.format(i+1,grid_points))
    Es[i] = vqe.expval(theta)
    i += 1

np.save('data/brute/grid.npy',x)
np.save('data/brute/parameters.npy',np.asarray(params))
np.save('data/brute/values{}.npy'.format(shots),Es)

