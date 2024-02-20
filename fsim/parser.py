# should change the filename
with open("../benchmark/opentimer_dump/c5315.txt", "r") as f:
    lines = f.read().split("\n")

# get the index of each section
start_pi_idx = lines.index("// Start PIs")
start_po_idx = lines.index("// Start POs")
start_wire_idx = lines.index("// Start wires")
start_cell_idx = lines.index("// Start cells")
start_digraph_idx = lines.index("digraph TimingGraph {")

# parse raw things
pis = lines[start_pi_idx + 1 : start_po_idx - 1]
pos = lines[start_po_idx + 1 : start_wire_idx - 1]
wires = lines[start_wire_idx + 1 : start_cell_idx - 1]
cells = lines[start_cell_idx + 1 : start_digraph_idx - 2]  # note: -2
connections = lines[start_digraph_idx + 1 : -2]  # note: -2

# pre-process raw things, still may contain unnecessary connections
pis = [x.split()[1][:-1] for x in pis]
pos = [x.split()[1][:-1] for x in pos]
wires = [x.split()[1][:-1] for x in wires]
cells = [(x.split()[1], x.split()[0]) for x in cells]
connections = [
    tuple(map(lambda y: y.strip(' ";'), x.split(" -> ")))
    for x in connections
    if "->" in x
]


def gate_enum_to_id(s):
    # ! IMPORTANT: note: should be the same as in C code
    gate_enum = ["INV", "AND", "OR", "XOR", "NAND", "NOR", "XNOR", "MUX", "CLKBUF"]
    for idx, enum in enumerate(gate_enum):
        if s.startswith(enum):
            return idx
    raise Exception(f"{s} is an unknown gate name")


# get the gate type of each cell
gate_type_ids = [gate_enum_to_id(x[1]) for x in cells]

# 1 to multiple relationship: 'GATE_NAME:OUT_PIN' -> {GATE_NAME:IN_PIN, ...}
connections_d = {}
for x in connections:
    if x[0].split(":")[0] == x[1].split(":")[0]:
        # same gate connection
        continue
    if x[0] not in connections_d:
        connections_d[x[0]] = set()
    connections_d[x[0]].add(x[1])

# GATE_NAME -> gate ID
gate_d = {y: idx for idx, y in enumerate(pis + pos + [x[0] for x in cells])}

# Only for c7552.v
# A, B, Z/ZN
# A1, A2, Z/ZN
# A1, A2, A3, Z/ZN
# A1, A2, A3, A4, Z/ZN
# A, B, S, Z/ZN
# -----
# Y/Z/ZN: 0, A/A1: 1, B/A2: 2, A3: 3, A4: 4, S: 5
# for MUX -> input hope to be in order of (A, B, S) = (1, 2, 5)
pin_d = {
    "Y": 0,
    "Z": 0,
    "ZN": 0,
    "A": 1,
    "A1": 1,
    "B": 2,
    "A2": 2,
    "A3": 3,
    "A4": 4,
    "S": 5,
}


def get_gate_and_pin_id(s):
    if ":" in s:
        gate_name, pin_name = s.split(":")
        return gate_d[gate_name], pin_d[pin_name]
    elif s in pis:
        return gate_d[s], pin_d["Z"]
    elif s in pos:
        return gate_d[s], pin_d["A"]
    else:
        raise Exception(f"{s} is an unknown gate:pin")


# output
print(len(pis))
print(len(pos))
print(len(cells))

print(len(connections_d))
for k, v in connections_d.items():
    print(
        " ".join(map(str, get_gate_and_pin_id(k))),
        len(v),
        *map(lambda x: " ".join(map(str, x)), map(get_gate_and_pin_id, v)),
    )

print(" ".join(map(str, gate_type_ids)))
