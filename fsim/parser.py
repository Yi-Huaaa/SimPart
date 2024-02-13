# should change the filename
with open("../benchmark/opentimer_dump/vga_lcd.txt", "r") as f:
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
    gate_enum = [
        "INV", "AND", "OR", "XOR", "NAND", "NOR", "XNOR", "MUX", "CLKBUF", # original
        # vga_lcd
        'AOI21', 'AOI211', 'AOI22', 'AOI221', 'AOI222',
        'BUF',
        'DFF', 'DFFR', 'DFFRS', 'DFFS',
        'HA',
        'OAI21', 'OAI211', 'OAI22', 'OAI221', 'OAI222', 'OAI33',
        'SDFF', 'SDFFR', 'SDFFRS', 'SDFFS',
    ]
    # NOTE: this is a hack: this `for` loop is to match the gates from vga_lcd, which don't have a number
    # after their gate type, so we can do an exact match
    for idx, enum in enumerate(gate_enum):
        if s.split("_")[0] == enum:
            return idx
    # NOTE: However, for the original gates, they can have a number after the gate type, e.g., NAND4_X2,
    # so we do a prefix match if the exact match fails
    for idx, enum in enumerate(gate_enum):
        if s.startswith(enum):
            return idx
    raise Exception(f"{s} is an unknown gate name")

# get the gate type of each cell
gate_type_ids = [gate_enum_to_id(x[1]) for x in cells]

# build gate name -> gate type mapping
gate_name_to_type_d = {x[0]: x[1].split("_")[0] for x in cells}

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
# pin_d = {"Y": 0, "Z": 0, "ZN": 0, "A": 1, "A1": 1, "B": 2, "A2": 2, "A3": 3, "A4": 4, "S": 5}


# vga_lcd
pin_d = {
    'INV':  ['ZN', 'A'],
    'AND2': ['ZN', 'A1', 'A2'],
    'AND3': ['ZN', 'A1', 'A2', 'A3'],
    'AND4': ['ZN', 'A1', 'A2', 'A3', 'A4'],
    'OR2':  ['ZN', 'A1', 'A2'],
    'OR3':  ['ZN', 'A1', 'A2', 'A3'],
    'OR4':  ['ZN', 'A1', 'A2', 'A3', 'A4'],
    'XOR2': ['ZN', 'A', 'B'],
    'XOR3': ['ZN', 'A1', 'A2', 'A3'],
    'XOR4': ['ZN', 'A1', 'A2', 'A3', 'A4'],
    'NAND2':['ZN', 'A1', 'A2'],
    'NAND3':['ZN', 'A1', 'A2', 'A3'],
    'NAND4':['ZN', 'A1', 'A2', 'A3', 'A4'],
    'NOR2': ['ZN', 'A1', 'A2'],
    'NOR3': ['ZN', 'A1', 'A2', 'A3'],
    'NOR4': ['ZN', 'A1', 'A2', 'A3', 'A4'],
    'XNOR2':['ZN', 'A', 'B'],
    'XNOR3':['ZN', 'A1', 'A2', 'A3'],
    'XNOR4':['ZN', 'A1', 'A2', 'A3', 'A4'],
    'MUX2': ['ZN', 'A', 'B', 'S'],
    'CLKBUF': ['Z', 'A'],

    'AOI21': ['ZN', 'A', 'B1', 'B2'],
    'AOI211': ['ZN', 'A', 'B', 'C1', 'C2'],
    'AOI22': ['ZN', 'A1', 'A2', 'B1', 'B2'],
    'AOI221': ['ZN', 'A', 'B1', 'B2', 'C1', 'C2'],
    'AOI222': ['ZN', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
    'DFF': ['QN', 'D', 'CK'],
    'DFFR': ['QN', 'RN', 'D', 'CK'], # Note: it have .Q && .QN, but in vga_lcd, it only have .QN
    'DFFS': ['QN', 'SN', 'D', 'CK'], # Note: it have .Q && .QN, but in vga_lcd, it only have .QN
    'HA': ['S', 'CO', 'B', 'A'], # in: A, B; out: S, CO
    'OAI21': ['ZN', 'B2', 'B1', 'A'],
    'OAI211': ['ZN', 'C2', 'B', 'C1', 'A'],
    'OAI22': ['ZN', 'B2', 'B1', 'A2', 'A1'],
    'OAI221': ['ZN', 'A', 'B1', 'B2', 'C1', 'C2', ], 
    'OAI222': ['ZN', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2'], # Note: 6 inputs # MAX number of inputs
    'OAI33': ['ZN', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'], # Note: 6 inputs # MAX number of inputs
    'SDFF': ['Q', 'SE', 'D', 'SI', 'CK'], # Note: it have .Q && .QN, but in vga_lcd, it only have .Q
}

pin_d = {k: {x: idx for idx, x in enumerate(v)} for k, v in pin_d.items()}
for k in pin_d:
    if 'Z' in pin_d[k]:
        pin_d[k]['ZN'] = pin_d[k]['Z']
    elif 'ZN' in pin_d[k]:
        pin_d[k]['Z'] = pin_d[k]['ZN']
    if 'Q' in pin_d[k]:
        pin_d[k]['QN'] = pin_d[k]['Q']
    elif 'QN' in pin_d[k]:
        pin_d[k]['Q'] = pin_d[k]['QN']

def get_gate_and_pin_id(s):
    if ":" in s:
        gate_name, pin_name = s.split(":")
        gate_type = gate_name_to_type_d[gate_name]
        # print('debug: ', gate_name, pin_name, gate_type)
        return gate_d[gate_name], pin_d[gate_type][pin_name]
    elif s in pis:
        return gate_d[s], 0 # output
    elif s in pos:
        return gate_d[s], 1 # input
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
