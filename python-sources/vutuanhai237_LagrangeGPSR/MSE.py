import base, constant
import numpy as np, qiskit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import importlib
importlib.reload(constant)
importlib.reload(base)

step_sizes = base.create_logsin_step_sizes(0.01, 3, 0.01)
thetas = np.asarray([np.pi/2, np.pi/3, np.pi/6])
def u(qc, thetas):
    qc.rx(thetas[0], 0)
    qc.rz(thetas[1], 1)
    qc.cry(thetas[2], 0, 1)
    return qc
def f(thetas):
    qc = qiskit.QuantumCircuit(2,2)
    qc = u(qc, thetas)
    return base.measure(qc, [0, 1])

true_grad = base.true_grad(thetas)

mean_MSE_stds = []
mean_MSE_finites = []
std_MSE_stds = []
std_MSE_finites = []
for step_size in step_sizes:
    print(step_size)
    grad_finites = []
    grad_stds = []

    for _ in range(0, 1000):
        grad_finite = []
        grad_std = []
        for j in range(0, thetas.shape[0]):
            length = thetas.shape[0]
            f_left = f(thetas + step_size * base.unit_vector(j, length))
            f_right = f(thetas + step_size * base.unit_vector(j, length))
            if j != 2:
                grad_std.append(base.a_pseudo_two_prx(f_left, f_right, step_size))
                # grad_finite.append(base.two_finite_diff(f, thetas_origin, j, step_size))
            else:
                grad_std.append(base.pseudo_four_prx(f, thetas, j))
            grad_finite.append(base.a_two_finite_diff(f_left, f_right, step_size))
        grad_stds.append(mean_squared_error(grad_std, true_grad))
        grad_finites.append(mean_squared_error(grad_finite, true_grad))
    mean_MSE_stds.append(np.mean(grad_stds,axis = 0))
    std_MSE_stds.append(np.std(grad_stds,axis = 0))

    mean_MSE_finites.append(np.mean(grad_finites,axis = 0))
    std_MSE_finites.append(np.std(grad_finites,axis = 0))

np.savetxt('mean_MSE_stds.txt', mean_MSE_stds)
np.savetxt('std_MSE_stds.txt', std_MSE_stds)
np.savetxt('mean_MSE_finites.txt', mean_MSE_finites)
np.savetxt('std_MSE_finites.txt', std_MSE_finites)