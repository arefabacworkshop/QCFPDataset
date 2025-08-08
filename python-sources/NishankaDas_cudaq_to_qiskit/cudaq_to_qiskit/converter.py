import ast
import os
import numpy as np
from qiskit import QuantumCircuit

GATE_MAPPING = {
    "h": "h",
    "x": "x",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "cx": "cx",
    "measure": "measure",
    "mz": "measure",
    "t": "t",
    "z": "z",
    "y": "y",
    "s": "s",
}

def eval_ast_expr(node):
    """Safely evaluate mathematical expressions from AST nodes."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = eval_ast_expr(node.left)
        right = eval_ast_expr(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "pi":
        return np.pi
    return None

def extract_operations_from_ast(node, operations):
    """Recursively extract quantum operations from AST nodes."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value
        if isinstance(func.func, ast.Name) and func.func.id in GATE_MAPPING:
            gate_name = func.func.id
            params = [eval_ast_expr(arg) for arg in func.args[:-1]]
            qubit = func.args[-1].id if isinstance(func.args[-1], ast.Name) else None
            operations.append((gate_name, params, qubit))

    for child in ast.iter_child_nodes(node):
        extract_operations_from_ast(child, operations)

def convert_cudaq_source_to_qiskit(cudaq_source):
    """Convert CUDA-Q function source code to a Qiskit circuit and Python code."""
    parsed_ast = ast.parse(cudaq_source)
    operations = []
    extract_operations_from_ast(parsed_ast, operations)
    qubits = sorted({op[2] for op in operations if op[2]})
    qubit_indices = {q: i for i, q in enumerate(qubits)}
    num_qubits = len(qubit_indices)
    qc = QuantumCircuit(num_qubits, num_qubits)
    qiskit_code = f"from qiskit import QuantumCircuit\nfrom qiskit.primitives import Sampler\nsampler = Sampler()\nqc = QuantumCircuit({num_qubits}, {num_qubits})\n"

    for gate, params, qubit in operations:
        q_idx = qubit_indices[qubit]
        qiskit_gate = GATE_MAPPING[gate]

        if qiskit_gate == "measure":
            qc.measure(q_idx, q_idx)
            qiskit_code += f"qc.measure({q_idx}, {q_idx})\n"
        elif params and all(p is not None for p in params):
            qiskit_code += f"qc.{qiskit_gate}({', '.join(map(str, params))}, {q_idx})\n"
            getattr(qc, qiskit_gate)(*params, q_idx)
        else:
            qiskit_code += f"qc.{qiskit_gate}({q_idx})\n"
            getattr(qc, qiskit_gate)(q_idx)

    qiskit_code += f"res = sampler.run(qc)\nprint(res.result())\n"
    return qc, qiskit_code
