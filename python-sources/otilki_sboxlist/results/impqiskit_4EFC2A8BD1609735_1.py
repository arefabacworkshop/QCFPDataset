# from : 00FF 0F0F 3333 5555 
#F[0] = X[0];
#F[1] = X[3];
#F[2] = X[2];
#F[3] = X[1];

from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
circuit = QuantumCircuit(4)

circuit.x(2)
Info_Op:  RNOT, Info_Line: 2, Op_Values  BS[0] 00ff  BS[1] 5555  BS[2] cccc  BS[3] 0f0f , Prev_Value 3333 , Rem Cost 100

circuit.ccx((1),(0),(3))
Info_Op:  CCNOT, Info_Line: 3, Op_Values  BS[0] 00ff  BS[1] 5555  BS[2] cccc  BS[3] 0f5a , Prev_Value 0f0f , Rem Cost 800

circuit.x(1)
Info_Op:  RNOT, Info_Line: 1, Op_Values  BS[0] 00ff  BS[1] aaaa  BS[2] cccc  BS[3] 0f5a , Prev_Value 5555 , Rem Cost 900

circuit.x(0)
Info_Op:  RNOT, Info_Line: 0, Op_Values  BS[0] ff00  BS[1] aaaa  BS[2] cccc  BS[3] 0f5a , Prev_Value 00ff , Rem Cost 1000

circuit.x(3)
Info_Op:  RNOT, Info_Line: 3, Op_Values  BS[0] ff00  BS[1] aaaa  BS[2] cccc  BS[3] f0a5 , Prev_Value 0f5a , Rem Cost 1100

circuit.ccx((2),(1),(0))
Info_Op:  CCNOT, Info_Line: 0, Op_Values  BS[0] 7788  BS[1] aaaa  BS[2] cccc  BS[3] f0a5 , Prev_Value ff00 , Rem Cost 1800

circuit.ccx((3),(2),(1))
Info_Op:  CCNOT, Info_Line: 1, Op_Values  BS[0] 7788  BS[1] aaaa  BS[2] cccc  BS[3] f0a5 , Prev_Value 6a2e , Rem Cost 2000 R

circuit.cx((3),(2))
Info_Op:  CNOT, Info_Line: 2, Op_Values  BS[0] 7788  BS[1] 6a2e  BS[2] cccc  BS[3] f0a5 , Prev_Value 3c69 , Rem Cost 1300 R

circuit.cx((0),(1))
Info_Op:  CNOT, Info_Line: 1, Op_Values  BS[0] 7788  BS[1] 6a2e  BS[2] 3c69  BS[3] f0a5 , Prev_Value 1da6 , Rem Cost 1100 R

circuit.cx((1),(2))
Info_Op:  CNOT, Info_Line: 2, Op_Values  BS[0] 7788  BS[1] 1da6  BS[2] 3c69  BS[3] f0a5 , Prev_Value 21cf , Rem Cost 900 R

circuit.ccx((3),(0),(1))
Info_Op:  CCNOT, Info_Line: 1, Op_Values  BS[0] 7788  BS[1] 1da6  BS[2] 21cf  BS[3] f0a5 , Prev_Value 6d26 , Rem Cost 700 R

print(circuit.depth()) #note that this is the non-decomposed depth
circuit.draw(output="mpl")
plt.show()
#X[0] = F[0];
#X[1] = F[3];
#X[2] = F[1];
#X[3] = F[2];

# to : 7788 F0A5 6D26 21CF 
# T-Depth : 4
# Depth : 33
