#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum circuit simulator module for QuantumSandbox.
Provides functionality to create, simulate, and analyze quantum circuits.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import io
import base64

class CircuitSimulator:
    """
    Class for simulating quantum circuits using Qiskit.
    """
    
    def __init__(self):
        """Initialize the circuit simulator with available backends."""
        self.statevector_backend = Aer.get_backend('statevector_simulator')
        self.qasm_backend = Aer.get_backend('qasm_simulator')
    
    def _create_circuit_from_definition(self, circuit_def):
        """
        Create a Qiskit QuantumCircuit from a circuit definition.
        
        Args:
            circuit_def (dict): Circuit definition with qubits and gates
            
        Returns:
            QuantumCircuit: A Qiskit quantum circuit
        """
        num_qubits = circuit_def.get('qubits', 1)
        gates = circuit_def.get('gates', [])
        
        # Create quantum circuit
        circuit = QuantumCircuit(num_qubits, num_qubits)
        
        # Add gates to the circuit
        for gate in gates:
            gate_type = gate.get('type', '').lower()
            
            if gate_type == 'h':
                for target in gate.get('targets', []):
                    circuit.h(target)
            
            elif gate_type == 'x':
                for target in gate.get('targets', []):
                    circuit.x(target)
            
            elif gate_type == 'y':
                for target in gate.get('targets', []):
                    circuit.y(target)
            
            elif gate_type == 'z':
                for target in gate.get('targets', []):
                    circuit.z(target)
            
            elif gate_type == 'cx' or gate_type == 'cnot':
                controls = gate.get('controls', [])
                targets = gate.get('targets', [])
                if controls and targets:
                    for control in controls:
                        for target in targets:
                            circuit.cx(control, target)
            
            elif gate_type == 'cz':
                controls = gate.get('controls', [])
                targets = gate.get('targets', [])
                if controls and targets:
                    for control in controls:
                        for target in targets:
                            circuit.cz(control, target)
            
            elif gate_type == 'ccx' or gate_type == 'toffoli':
                controls = gate.get('controls', [])
                targets = gate.get('targets', [])
                if len(controls) >= 2 and targets:
                    for target in targets:
                        circuit.ccx(controls[0], controls[1], target)
            
            elif gate_type == 'rx':
                targets = gate.get('targets', [])
                theta = gate.get('theta', 0)
                for target in targets:
                    circuit.rx(theta, target)
            
            elif gate_type == 'ry':
                targets = gate.get('targets', [])
                theta = gate.get('theta', 0)
                for target in targets:
                    circuit.ry(theta, target)
            
            elif gate_type == 'rz':
                targets = gate.get('targets', [])
                theta = gate.get('theta', 0)
                for target in targets:
                    circuit.rz(theta, target)
            
            elif gate_type == 's':
                for target in gate.get('targets', []):
                    circuit.s(target)
            
            elif gate_type == 'sdg':
                for target in gate.get('targets', []):
                    circuit.sdg(target)
            
            elif gate_type == 't':
                for target in gate.get('targets', []):
                    circuit.t(target)
            
            elif gate_type == 'tdg':
                for target in gate.get('targets', []):
                    circuit.tdg(target)
            
            elif gate_type == 'swap':
                targets = gate.get('targets', [])
                if len(targets) >= 2:
                    circuit.swap(targets[0], targets[1])
            
            elif gate_type == 'measure':
                for i, target in enumerate(gate.get('targets', [])):
                    circuit.measure(target, target)
        
        # Add measurements if not already added
        if not any(gate.get('type', '').lower() == 'measure' for gate in gates):
            circuit.measure_all()
            
        return circuit
    
    def simulate(self, circuit_def, shots=1024):
        """
        Simulate a quantum circuit and return the results.
        
        Args:
            circuit_def (dict): Circuit definition
            shots (int): Number of simulation shots
            
        Returns:
            dict: Simulation results including counts and statevector
        """
        circuit = self._create_circuit_from_definition(circuit_def)
        
        # Get circuit diagram
        circuit_diagram = circuit.draw(output='text').data
        
        # Run statevector simulation
        statevector_job = execute(circuit, self.statevector_backend)
        statevector_result = statevector_job.result()
        statevector = statevector_result.get_statevector(circuit)
        
        # Run measurement simulation
        measurement_circuit = circuit.copy()
        if not any(gate.get('type', '').lower() == 'measure' for gate in circuit_def.get('gates', [])):
            measurement_circuit.measure_all()
        
        qasm_job = execute(measurement_circuit, self.qasm_backend, shots=shots)
        qasm_result = qasm_job.result()
        counts = qasm_result.get_counts(circuit)
        
        # Generate histogram plot
        plt.figure(figsize=(10, 6))
        plot_histogram(counts)
        histogram_buf = io.BytesIO()
        plt.savefig(histogram_buf, format='png')
        plt.close()
        histogram_buf.seek(0)
        histogram_img = base64.b64encode(histogram_buf.read()).decode('utf-8')
        
        # Format statevector for JSON
        formatted_statevector = []
        for i, amplitude in enumerate(statevector):
            binary = format(i, f'0{circuit_def.get("qubits", 1)}b')
            formatted_statevector.append({
                "state": binary,
                "real": float(amplitude.real),
                "imag": float(amplitude.imag),
                "probability": float(abs(amplitude)**2)
            })
        
        return {
            "counts": counts,
            "statevector": formatted_statevector,
            "circuit_diagram": circuit_diagram,
            "histogram_image": histogram_img,
            "num_qubits": circuit_def.get('qubits', 1),
            "shots": shots
        }
    
    def export_to_qiskit(self, circuit_def):
        """
        Export a circuit definition to Qiskit Python code.
        
        Args:
            circuit_def (dict): Circuit definition
            
        Returns:
            str: Qiskit Python code
        """
        num_qubits = circuit_def.get('qubits', 1)
        gates = circuit_def.get('gates', [])
        
        code_lines = [
            "from qiskit import QuantumCircuit, Aer, execute",
            "from qiskit.visualization import plot_histogram",
            "",
            f"# Create a quantum circuit with {num_qubits} qubits",
            f"circuit = QuantumCircuit({num_qubits}, {num_qubits})",
            ""
        ]
        
        # Add gates to the code
        for gate in gates:
            gate_type = gate.get('type', '').lower()
            
            if gate_type == 'h':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.h({target})  # Hadamard gate on qubit {target}")
            
            elif gate_type == 'x':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.x({target})  # X gate (NOT) on qubit {target}")
            
            elif gate_type == 'y':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.y({target})  # Y gate on qubit {target}")
            
            elif gate_type == 'z':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.z({target})  # Z gate on qubit {target}")
            
            elif gate_type == 'cx' or gate_type == 'cnot':
                controls = gate.get('controls', [])
                targets = gate.get('targets', [])
                if controls and targets:
                    for control in controls:
                        for target in targets:
                            code_lines.append(f"circuit.cx({control}, {target})  # CNOT gate with control={control}, target={target}")
            
            elif gate_type == 'cz':
                controls = gate.get('controls', [])
                targets = gate.get('targets', [])
                if controls and targets:
                    for control in controls:
                        for target in targets:
                            code_lines.append(f"circuit.cz({control}, {target})  # CZ gate with control={control}, target={target}")
            
            elif gate_type == 'ccx' or gate_type == 'toffoli':
                controls = gate.get('controls', [])
                targets = gate.get('targets', [])
                if len(controls) >= 2 and targets:
                    for target in targets:
                        code_lines.append(f"circuit.ccx({controls[0]}, {controls[1]}, {target})  # Toffoli gate")
            
            elif gate_type == 'rx':
                targets = gate.get('targets', [])
                theta = gate.get('theta', 0)
                for target in targets:
                    code_lines.append(f"circuit.rx({theta}, {target})  # RX rotation by {theta} on qubit {target}")
            
            elif gate_type == 'ry':
                targets = gate.get('targets', [])
                theta = gate.get('theta', 0)
                for target in targets:
                    code_lines.append(f"circuit.ry({theta}, {target})  # RY rotation by {theta} on qubit {target}")
            
            elif gate_type == 'rz':
                targets = gate.get('targets', [])
                theta = gate.get('theta', 0)
                for target in targets:
                    code_lines.append(f"circuit.rz({theta}, {target})  # RZ rotation by {theta} on qubit {target}")
            
            elif gate_type == 's':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.s({target})  # S gate on qubit {target}")
            
            elif gate_type == 'sdg':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.sdg({target})  # S dagger gate on qubit {target}")
            
            elif gate_type == 't':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.t({target})  # T gate on qubit {target}")
            
            elif gate_type == 'tdg':
                for target in gate.get('targets', []):
                    code_lines.append(f"circuit.tdg({target})  # T dagger gate on qubit {target}")
            
            elif gate_type == 'swap':
                targets = gate.get('targets', [])
                if len(targets) >= 2:
                    code_lines.append(f"circuit.swap({targets[0]}, {targets[1]})  # SWAP gate between qubits {targets[0]} and {targets[1]}")
        
        # Add measurements
        code_lines.append("")
        code_lines.append("# Add measurements")
        code_lines.append("circuit.measure_all()")
        
        # Add simulation code
        code_lines.extend([
            "",
            "# Draw the circuit",
            "circuit.draw(output='mpl')",
            "",
            "# Simulate the circuit",
            "simulator = Aer.get_backend('qasm_simulator')",
            "job = execute(circuit, simulator, shots=1024)",
            "result = job.result()",
            "counts = result.get_counts(circuit)",
            "",
            "# Plot the results",
            "plot_histogram(counts)"
        ])
        
        return "\n".join(code_lines)
