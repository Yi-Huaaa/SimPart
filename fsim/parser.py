import sys

# python parser.py <input> <output> <mode>

# should change the filename
with open(sys.argv[1], "r") as f:
    lines = f.read().split("\n")
outfile = open(sys.argv[2], "w")
mode = sys.argv[3]

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

pin_d = {
    'CLKBUF':{'A':  1, 'Z':  0},
    'INV':   {'A':  1, 'ZN': 0},
    'AND2':  {'A':  1, 'B':  2, 'Y': 0},
    'NAND2': {'A1': 1, 'A2': 2, 'ZN': 0},
    'NAND3': {'A1': 1, 'A2': 2, 'A3': 3, 'ZN': 0},
    'NAND4': {'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'ZN': 0},
    'NOR2':  {'A1': 1, 'A2': 2, 'ZN': 0},
    'NOR3':  {'A1': 1, 'A2': 2, 'A3': 3, 'ZN': 0},
    'NOR4':  {'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'ZN': 0},
    'AOI21': {'A':  1, 'B1': 2, 'B2': 3, 'ZN': 0}, # change to NAND3
    'AOI22': {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'ZN': 0}, # change to NAND4
    'DFF':   {'D':  1, 'CK': 2, 'Q':  0}, # change to NAND2
    'OAI21': {'A':  1, 'B1': 2, 'B2': 3, 'ZN': 0}, # change to NOR3
    'OAI22': {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'ZN': 0}, # change to NOR4
    'AND2X1':{'A': 1,  'B': 2,  'Y': 0},  # Add this line
    'INVX1': {'A': 1,  'Y': 0},  # Add this line
    'NAND3X1': {'A': 1, 'B': 2, 'C': 3, 'Y': 0},
}

# IMPORTANT: note: for unchanged gates, their order should be the same as in C code
gate_enum_d = {
    'INV': 0,
    'AND': 1,
    'OR': 2,
    'XOR': 3,
    'NAND': 4,
    'NOR': 5,
    'XNOR': 6,
    'MUX': 7,
    'CLKBUF': 8,
    'AOI21': 4, # change to NAND3
    'AOI22': 4, # change to NAND4
    'DFF': 4, # change to NAND2
    'OAI21': 5, # change to NOR3
    'OAI22': 5, # change to NOR4
}

# a mapping from gate name to the key that will be used to lookup the pin_d table
gate_name_to_pin_d_key = {x[0]: x[1].split('_')[0] for x in cells}

# def gate_enum_to_id(s):
#     for enum in gate_enum_d:
#         if s.startswith(enum):
#             return gate_enum_d[enum]
#     raise Exception(f"{s} is an unknown gate name")
def gate_enum_to_id(s):
    for enum in gate_enum_d:
        if s.startswith(enum):
            return gate_enum_d[enum]
    # If gate name is unknown, return AND gate with ID 1
    print(f"Warning: {s} is an unknown gate name. Defaulting to AND (ID 1).")
    return 1  # Return AND gate ID as default


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
# pin_d = {
#     "Y": 0,
#     "Z": 0,
#     "ZN": 0,
#     "A": 1,
#     "A1": 1,
#     "B": 2,
#     "A2": 2,
#     "A3": 3,
#     "A4": 4,
#     "S": 5,
# }

# def get_gate_and_pin_id(s):
#     if ":" in s:
#         gate_name, pin_name = s.split(":")
#         pin_d_key = gate_name_to_pin_d_key[gate_name]
#         return gate_d[gate_name], pin_d[pin_d_key][pin_name]
#     elif s in pis:
#         return gate_d[s], 0
#     elif s in pos:
#         return gate_d[s], 1
#     else:
#         raise Exception(f"{s} is an unknown gate:pin")
def get_gate_and_pin_id(s):
    if ":" in s:
        gate_name, pin_name = s.split(":")
        # Check if gate_name has a corresponding entry in gate_name_to_pin_d_key
        if gate_name in gate_name_to_pin_d_key:
            pin_d_key = gate_name_to_pin_d_key[gate_name]
            # Check if pin_d_key exists in pin_d, otherwise provide default
            if pin_d_key in pin_d and pin_name in pin_d[pin_d_key]:
                return gate_d[gate_name], pin_d[pin_d_key][pin_name]
            else:
                print(f"Warning: {pin_name} for {gate_name} not found in pin_d. Using default.")
                return gate_d[gate_name], 0  # Default pin ID 0
        else:
            print(f"Warning: {gate_name} not found in gate_name_to_pin_d_key. Using default pin mapping.")
            return gate_d[gate_name], 0  # Default pin ID 0
    elif s in pis:
        return gate_d[s], 0
    elif s in pos:
        return gate_d[s], 1
    else:
        raise Exception(f"{s} is an unknown gate:pin")


# remove loops
from collections import deque

source = deque(pis)
visited = set()

while source:
    if mode == 'bfs':
        cur = source.popleft()
    elif mode == 'dfs':
        cur = source.pop()
    else:
        raise Exception('unknown mode')
    
    if cur in visited:
        continue
    visited.add(cur)

    # debug
    if len(visited) % 100000 == 0:
        print('progress', len(visited))

    for post in ['', ':Z', ':ZN', ':Q', ':Y', ':QN']:
        if cur + post in connections_d:
            outputs = connections_d[cur + post]
            break
    else:
        if cur in pos:
            continue
        print(f"Warning: No connection found for {cur}. Skipping this node.")
        continue  # Skip nodes with no connections instead of raising an exception
    # else:
    #     if cur in pos:
    #         continue
    #     print(f"Failed to find connection for: {cur}")
    #     raise Exception('not found!')
    #     # if cur in pos:
    #     #     continue
    #     # raise Exception('not found!')

    outputs = [x.split(':')[0] for x in outputs]
    for out in outputs:
        if out in visited:
            connections_d[cur + post] = {
                x for x in connections_d[cur + post]
                if not x.startswith(out)
            }
        else:
            source.append(out)

# output
print(len(pis), file=outfile)
print(len(pos), file=outfile)
print(len(cells), file=outfile)

print(len(connections_d), file=outfile)
for k, v in connections_d.items():
    print(
        " ".join(map(str, get_gate_and_pin_id(k))),
        len(v),
        *map(lambda x: " ".join(map(str, x)), map(get_gate_and_pin_id, v)),
        file=outfile
    )

print(" ".join(map(str, gate_type_ids)), file=outfile)

# def find(s):
#     s = s.split(':')[0]
#     result = []
#     for k, v in connections_d.items():
#         for vv in v:
#             if s in vv: result.append(k)
#     return result

# cur = 'ris_22_'
# seen = set()
# while True:
#     if ':' in cur:
#         gate_name = cur.split(":")[0]
#         pin_d_key = gate_name_to_pin_d_key[gate_name]
#     else:
#         pin_d_key = None
#     print('cur', cur, get_gate_and_pin_id(cur), pin_d_key)

#     if cur.startswith('in'):
#         break
    
#     assert cur not in seen, 'loop!'
#     seen.add(cur)
    
#     l = find(cur)
#     assert l, 'not l'
#     cur = l[0]