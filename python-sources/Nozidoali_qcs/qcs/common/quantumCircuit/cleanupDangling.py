from .base import QuantumCircuit

def cleanup_dangling_hadamard(self) -> QuantumCircuit:
    _circuit = QuantumCircuit()
    _circuit.request_qubits(self.n_qubits)
    is_had: dict[int, bool] = {i: False for i in range(self.n_qubits)}
    for i, gate in enumerate(self.gates):
        if gate["name"] == "HAD":
            is_had[gate["target"]] = not is_had[gate["target"]]
        else:
            for j in QuantumCircuit.deps_of(gate):
                if is_had[j]:
                    _circuit.add_gate({"name": "HAD", "target": j})
                    is_had[j] = False
            _circuit.add_gate(gate)
    for j in range(self.n_qubits):
        if is_had[j]:
            _circuit.add_gate({"name": "HAD", "target": j})
            is_had[j] = False
    return _circuit

def apply_rule_based_rewriting(
    circ: QuantumCircuit,
    rules: list[callable],
) -> tuple[QuantumCircuit, int]:
    g = circ.gates
    out: list[dict] = []
    i = 0
    applied = 0
    while i < len(g):
        for rule in rules:
            skip, replacement, did_apply = rule(g, i)
            if skip > 0:
                out.extend(replacement)
                i += skip
                if did_apply:
                    applied += 1
                break
        else:
            out.append(g[i])
            i += 1

    new_circ = QuantumCircuit()
    new_circ.n_qubits = circ.n_qubits
    new_circ.gates = out
    return new_circ, applied

def commutes(g1: dict, g2: dict) -> bool:
    return QuantumCircuit.deps_of(g1).isdisjoint(QuantumCircuit.deps_of(g2))

def cancel_double_x(g: list[dict], i: int) -> tuple[int, list[dict], bool]:
    if g[i]["name"] != "X":
        return 0, [], False

    tgt = g[i]["target"]
    commute_buffer = []
    for j in range(i + 1, min(len(g), i + 10)):
        gj = g[j]
        if gj["name"] == "X" and gj["target"] == tgt:
            # Can cancel with g[i]; drop both and preserve commuted gates
            return j - i + 1, commute_buffer, True
        elif commutes(g[i], gj):
            commute_buffer.append(gj)
        else:
            break

    return 0, [], False

def toffoli_cancel_or_rewrite(g: list[dict], i: int) -> tuple[int, list[dict], bool]:
    if g[i]["name"] != "Tof":
        return 0, [], False

    a = g[i]
    ctrl1, ctrl2, tgt = a["ctrl1"], a["ctrl2"], a["target"]
    deps = QuantumCircuit.deps_of(a)

    x_on_ctrl1 = 0
    x_on_ctrl2 = 0
    junk = []
    j = i + 1

    while j < len(g):
        gj = g[j]

        if gj["name"] == "Tof":
            if gj["ctrl1"] == ctrl1 and gj["ctrl2"] == ctrl2 and gj["target"] == tgt:
                # Matching Toffoli found
                if x_on_ctrl1 == 0 and x_on_ctrl2 == 0:
                    return j - i + 1, junk, True
                elif x_on_ctrl1 + x_on_ctrl2 == 1:
                    ctrl = ctrl1 if x_on_ctrl1 == 1 else ctrl2
                    return j - i + 1, junk + [
                        {"name": "CNOT", "ctrl": ctrl, "target": tgt}, 
                        {"name": "X", "target": ctrl},
                    ], True
                elif x_on_ctrl1 == 1 and x_on_ctrl2 == 1:
                    return j - i + 1, junk + [
                        {"name": "CNOT", "ctrl": ctrl1, "target": tgt},
                        {"name": "X", "target": tgt},
                        {"name": "CNOT", "ctrl": ctrl2, "target": tgt},
                        {"name": "X", "target": ctrl1},
                        {"name": "X", "target": ctrl2},
                    ], True
                else:
                    break  # more than 2 Xs — undefined
            else:
                break  # incompatible Toffoli
        elif QuantumCircuit.deps_of(gj).isdisjoint(deps):
            junk.append(gj)
            j += 1
        elif gj["name"] == "X":
            if gj["target"] == ctrl1:
                x_on_ctrl1 ^= 1
            elif gj["target"] == ctrl2:
                x_on_ctrl2 ^= 1
            j += 1
        else:
            break
    return 0, [], False


def cancel_double_toffoli(g: list[dict], i: int) -> tuple[int, list[dict], bool]:
    if g[i]["name"] != "Tof":
        return 0, [], False
    deps_i = QuantumCircuit.deps_of(g[i])
    ctrl_pair = {g[i]["ctrl1"], g[i]["ctrl2"]}
    tgt = g[i]["target"]
    j = i + 1
    junk = []
    while j < len(g):
        gate_j = g[j]
        if gate_j["name"] == "Tof":
            if {gate_j["ctrl1"], gate_j["ctrl2"]} == ctrl_pair and gate_j["target"] == tgt:
                return j - i + 1, junk, True
            else:
                break  # another Toffoli that interferes

        if QuantumCircuit.deps_of(gate_j).isdisjoint(deps_i):
            junk.append(gate_j)
            j += 1
        else:
            break
    return 0, [], False

def cleanup(circ: QuantumCircuit) -> QuantumCircuit:
    rules = [
        cancel_double_toffoli,
        cancel_double_x,
        toffoli_cancel_or_rewrite
    ]
    circ = cleanup_dangling_hadamard(circ)
    while True:
        circ, applied = apply_rule_based_rewriting(circ, rules)
        if applied == 0:
            return circ
