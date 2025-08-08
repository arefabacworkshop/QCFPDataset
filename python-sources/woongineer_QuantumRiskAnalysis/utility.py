from typing import Optional

from qiskit import QuantumCircuit, transpile


def get_min_cx_brute(qcircuit: QuantumCircuit,
                     backend: Optional = None,
                     num_of_shot: Optional[int] = 1
                     ) -> int:
    """Get min number of CX gate by brute-force

    Args:
        qcircuit: quantum circuit to analyse
        backend: backend provider
        num_of_shot: num of trial

    Returns:
        min number of CX gates
    """
    if not backend:
        backend = None

    num_of_shot_list = []
    for i in range(num_of_shot):
        num_of_cx = transpile(qcircuit.decompose(reps=100), backend=backend).count_ops()['cx']
        num_of_shot_list.append(num_of_cx)

    return min(num_of_shot_list)