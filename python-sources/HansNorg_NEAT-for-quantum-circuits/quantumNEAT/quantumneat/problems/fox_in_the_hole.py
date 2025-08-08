from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import numpy as np
from quantumneat.configuration import QuantumNEATConfig
from quantumneat.genome import Genome
from qulacs import QuantumState
from qiskit.circuit import Parameter

from quantumneat.quant_lib_np import dtype, sz, Id, Z
from quantumneat.problem import Problem
from quantumneat.problems.fox_in_a_hole_gym import FoxInAHolev2

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit

class FoxInTheHole(Problem):
    def __init__(self, config: QuantumNEATConfig, n_holes = 5, len_state = 2, max_steps = 6, **kwargs) -> None:
        super().__init__(config)
        self.env = FoxInAHoleExact(n_holes, max_steps, len_state)

    def fitness(self, genome:Genome) -> float:
        circuit = genome.get_circuit()[0]
        parameters = genome.get_parameters()
        fitness = self.env.max_guesses - self.energy(circuit, parameters)
        return fitness
    
    def energy(self, circuit:Circuit, parameters, no_optimization=False) -> float:
        memory = self.env.reset()
        done = False
        while not done:
            action = self.get_action(circuit, memory)
            memory, avg_steps, done, _ = self.env.step(action)
        return avg_steps
    
    def add_encoding_layer(self, circuit:Circuit):
        if self.config.simulator == "qiskit":
            circuit.rx(Parameter('enc_0'), 0)
            circuit.rx(Parameter('enc_1'), 1)
        elif self.config.simulator == "qulacs":
            circuit.add_parametric_RX_gate(0, -1)
            circuit.add_parametric_RX_gate(1, -1)
    
    def solution(self) -> float:
        return brute_force_fith(self.env.n_holes, self.env.max_guesses)
    
    def get_action(self, circuit:Circuit, memory):
        if self.config.simulator == 'qulacs':
            for i, param in enumerate(memory):
                circuit.set_parameter(i, param)

            state = QuantumState(self.config.n_qubits)
            circuit.update_quantum_state(state)
            psi = state.get_vector()
            # operator = Zs(n_holes, len_state)
            # expval = (np.conj(psi).T @ operator @ psi).real
            # print(expval)

            expvals = []
            for i in range(self.env.n_holes):
            # for i in range(len_state, 0, -1):
                operator = Z(i, self.config.n_qubits)
                expval = (np.conj(psi).T @ operator @ psi).real
                expvals.append(expval)

            # print(expvals)
            action = np.argmax(expvals)
            return action
        else:
            raise NotImplementedError(f"Simulator type {self.config.simulator} not implemented.")

class FoxInTheHoleNGates(FoxInTheHole):
    def fitness(self, genome: Genome) -> float:
        return super().fitness(genome)+1/(len(genome.genes)+1)
    
def Zs(n_Zs, n_qubits):
    U = np.array([1], dtype = dtype)
    for _ in range(n_Zs):
        U = np.kron(U, sz)
    for _ in range(n_Zs, n_qubits):
        U = np.kron(U, Id)    
    return U

def run_episode(circuit:Circuit, config:QuantumNEATConfig):
    n_holes = 5
    len_state = 2
    max_steps = 6
    env = FoxInAHolev2(n_holes, max_steps, len_state)
    env_state = env.reset()
    return_ = 1
    for _ in range(max_steps):
        env_state, reward, done = choose_action(circuit, env, env_state, len_state, n_holes, config.n_qubits)
        if done:
            break
        return_ += 1 #reward
    # print(returns)
    return return_

def get_multiple_returns(circuit, config, n_iterations = 100):
    returns = []
    for i in range(0, n_iterations):
        returns.append(run_episode(circuit, config))
    return returns

def choose_action(circuit:Circuit, env:FoxInAHolev2, env_state, len_state, n_holes, n_qubits):
    for i, param in enumerate(env_state):
        circuit.set_parameter(i, param)

    state = QuantumState(n_qubits)
    circuit.update_quantum_state(state)
    psi = state.get_vector()
    # operator = Zs(n_holes, len_state)
    # expval = (np.conj(psi).T @ operator @ psi).real
    # print(expval)

    expvals = []
    for i in range(n_holes):
    # for i in range(len_state, 0, -1):
        operator = Z(i, n_qubits)
        expval = (np.conj(psi).T @ operator @ psi).real
        expvals.append(expval)

    # print(expvals)
    action = np.argmax(expvals)
    # print(action)
    env_state, reward, done, _ = env.step(action)
    # print(env_state)
    # print(reward)
    # print(done)
    return env_state, reward, done

def add_encoding_layer(config:QuantumNEATConfig, circuit:Circuit):
    # add_h_layer(config, circuit)
    if config.simulator == "qiskit":
        circuit.rx(Parameter('enc_0'), 0)
        circuit.rx(Parameter('enc_1'), 1)
    elif config.simulator == "qulacs":
        circuit.add_parametric_RX_gate(0, -1)
        circuit.add_parametric_RX_gate(1, -1)

class FoxInAHoleExact:
    def __init__(self, n_holes, max_guesses, memory_length = 2) -> None:
        self.n_holes = n_holes
        self.max_guesses = max_guesses
        self.transfermatrix = np.array([[0.5 if abs(i-j) == 1 else 0 for i in range(n_holes)] for j in range(n_holes)])
        self.transfermatrix[1,0] = 1
        self.transfermatrix[-2, -1] = 1
        self.initial_state = np.ones(n_holes)/n_holes 
        self.memory_length = memory_length
    
    def reset(self):
        """
        Resets the environment.

        Returns
        -------
            memory (np.array): numpy array with the last two made guesses.
        """
        # reset the environment to initial random state
        self.state = self.initial_state.copy()
        self.memory = -1*np.ones(int(self.memory_length)) #n previous picks encoding TODO talk to Jesse about this (angles are mod 2pi, does this make sense?)
        self.guess_counter = 0
        self.avg_steps = 0
        return self.memory
    
    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. Computes the average return it would get from this step.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            memory (np.array): numpy array with the last two made guesses (inlcuding the guess made with this functions call).
            avg_reward (int): the average reward for the guess.
            done (bool): True if the game is finished, False if not.
            {} (?): Ignore this, its a remnant of the gym environment guidelines. 

        """
        done = self._update_state(action)
        self.memory = np.roll(self.memory, 1)
        self.memory[0] = action
        return self.memory, self.avg_steps, done, {}
    
    def _update_state(self, action):
        self.guess_counter += 1
        self.avg_steps += self.state[action]*(self.guess_counter)
        self.state[action] = 0
        self.state = np.inner(self.transfermatrix,self.state)
        if self.guess_counter == self.max_guesses:
            self.avg_steps += np.sum(self.state)*(self.guess_counter + 1)
            return True
        return False
    
    def multiple_steps(self, actions:np.ndarray):
        self.reset()
        for action in actions:
            self._update_state(action)
        return self.avg_steps

def brute_force_fith(n_holes, max_guesses, do_print = False):
    def configurations(N):
        if N == 1:
            for i in range(n_holes):
                yield [i]
        else:
            for configuration in configurations(N-1):
                for i in range(n_holes):
                    yield np.concatenate(([i], configuration))    

    avgs = {}
    starttime1 = time()
    env = FoxInAHoleExact(n_holes, max_guesses)
    for ind, guesses in enumerate(configurations(max_guesses)):
        avgs[ind] = env.multiple_steps(guesses)
        if do_print and ind%10000 == 0:
            print(f"Strategy: {ind}", end="\r")
    if do_print:
        print(end="                                                                                                                       \r")
    runtime1 = time()-starttime1
    
    if do_print:
        min_performance = np.inf
        min_configuration = []
        for i, configuration in enumerate(configurations(max_guesses)):
            if avgs[i] < min_performance:
                min_performance = avgs[i]
                min_configuration = configuration
        print(f"{max_guesses}: {min_performance:.2f} {min_configuration}, {runtime1:.4f}, {ind}")
    else:
        min_performance = min(avgs)
    # print(f"{min_performance}")
    return min_performance

if __name__ == "__main__":
    brute_force_fith(5, 6)
    # brute_force_fith(5, 10)
    exit()

    from qulacs import ParametricQuantumCircuit
    from quantumneat.configuration import QuantumNEATConfig
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    config = QuantumNEATConfig(5, 10)
    circuit_ = ParametricQuantumCircuit(config.n_qubits)
    params = [1.1, 2.1, .4, -0.1, -2.3]
    add_encoding_layer(config, circuit_)
    for i in range(config.n_qubits):
        circuit_.add_parametric_RX_gate(i, params[i])
    circuit_.add_CNOT_gate(0, 1)
    circuit_.add_CNOT_gate(1, 2)
    circuit_.add_CNOT_gate(2, 3)
    circuit_.add_CNOT_gate(3, 4)
    circuit_.add_CNOT_gate(4, 0)
    # print(circuit)

    amounts = [1, 10, 100, 1000, 10000]
    # amounts = [1, 10, 100, 1000]
    # amounts = [1, 2, 3, 4, 5]
    # amounts = [1,2]
    repeats = 100
    save = False

    return_data = pd.DataFrame()
    time_data = pd.DataFrame()
    for i in amounts:
        mean_returns = []
        runtimes = []
        for j in range(repeats):
            starttime = time()
            returns = get_multiple_returns(circuit_, config, i)
            runtime = time()-starttime
            runtimes.append(runtime)
            mean_returns.append(np.mean(returns))
        return_data[i] = mean_returns
        time_data[i] = runtimes
        print(f"Return of {i:5} iterations: mean = {np.mean(mean_returns)}, std = {np.std(mean_returns)}, mean time = {np.mean(runtimes)}")

    # import pickle
    if save:
        np.save('return_data', return_data, allow_pickle=True)
        np.save('time_data', time_data, allow_pickle=True)
    # sns.boxplot(return_data)
    # plt.title("Returns")
    # plt.show()
    # # plt.savefig("plot1.png")
    # plt.close()
    # plt.title("Returns")
    # sns.boxplot(return_data)
    # plt.xscale("log")
    # plt.savefig("plot2.png")
    # plt.close()
    # sns.boxplot(time_data)
    # plt.title("Time")
    # # plt.savefig("plot3.png")
    # # plt.close()
    # # sns.boxplot(time_data)
    # # plt.title("Time")
    # # plt.xscale("log")
    # # plt.savefig("plot4.png")
    # # plt.close()
    # # sns.boxplot(time_data)
    # # plt.title("Time")
    # plt.yscale("log")
    # plt.show()
    # plt.savefig("plot5.png")
    # plt.close()
    # sns.boxplot(time_data)
    # plt.title("Time")
    # plt.xscale("log")
    # plt.xscale("linear")
    # # plt.yscale("log")
    # plt.savefig("plot6.png")
    # plt.close()

    return_opt = pd.DataFrame()
    time_opt = pd.DataFrame()
    for i in amounts:
        mean_returns = []
        runtimes = []
        for j in range(repeats):
            starttime = time()
            def objective(params):
                for ind, param in enumerate(params):
                    circuit_.set_parameter(ind+2, param)
                return get_multiple_returns(circuit_, config, i)
            returns = minimize(objective,params, method="COBYLA", tol=1e-4, options={'maxiter':config.optimize_energy_max_iter}).fun
            runtime = time()-starttime
            runtimes.append(runtime)
            mean_returns.append(np.mean(returns))
        return_opt[i] = mean_returns
        time_opt[i] = runtimes
        print(f"Return of {i:5} iterations: mean = {np.mean(mean_returns)}, std = {np.std(mean_returns)}, mean time = {np.mean(runtimes)}")

    if save:
        np.save('return_opt', return_opt, allow_pickle=True)
        np.save('time_opt', time_opt, allow_pickle=True)
    # sns.boxplot(return_data)
    # plt.title("Returns")
    # plt.show()
    # sns.boxplot(time_data)
    # plt.title("Time")
    # plt.yscale("log")
    # plt.show()
    # sns.boxplot(return_data, color='red')
    # sns.boxplot(return_opt, color='green')
    # plt.show()
    print(return_data.head())
    data = pd.DataFrame()
    data['type'] = np.concatenate((['no_optimization' for _ in range(repeats*len(amounts))], ['optimization' for _ in range(repeats*len(amounts))])) 
    amounts_part = np.ravel([[i for _ in range(repeats)] for i in amounts])
    data['amounts'] = np.concatenate((amounts_part, amounts_part))
    data['returns'] = np.concatenate((np.ravel([return_data[i] for i in amounts]), np.ravel([return_opt[i] for i in amounts])))
    data['times'] = np.concatenate((np.ravel([time_data[i] for i in amounts]),np.ravel([time_opt[i] for i in amounts])))
    print(data.head())
    sns.boxplot(data=data, x='amounts', y='returns', hue='type')
    plt.title("Returns")
    plt.savefig("returns.png")
    plt.close()
    plt.title("Time")
    sns.boxplot(data=data, x='amounts', y='times', hue='type')
    plt.savefig("times.png")
    plt.close()