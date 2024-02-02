with open('../benchmark/opentimer_dump/ring1.txt', 'r') as f:
    lines = f.read().split('\n')

start_pi_idx = lines.index('// Start PIs')
start_po_idx = lines.index('// Start POs')
start_wire_idx = lines.index('// Start wires')
start_cell_idx = lines.index('// Start cells')
start_digraph_idx = lines.index('digraph TimingGraph {')

pis = lines[start_pi_idx+1:start_po_idx-1]
pos = lines[start_po_idx+1:start_wire_idx-1]
wires = lines[start_wire_idx+1:start_cell_idx-1]
cells = lines[start_cell_idx+1:start_digraph_idx-2] # note: -2
connections = lines[start_digraph_idx+1:-2] # note: -2

pis = [x.split()[1][:-1] for x in pis]
pos = [x.split()[1][:-1] for x in pos]
wires = [x.split()[1][:-1] for x in wires]
cells = [
    (x.split()[1], x.split()[0])
    for x in cells
]
connections = [
    tuple(map(lambda y: y.strip(' ";'), x.split(' -> ')))
    for x in connections if '->' in x
]

def gate_enum_to_id(s):
    # note: should be the same as in C code
    gate_enum = ['INV', 'AND', 'OR', 'XOR', 'NAND', 'NOR']
    for idx, enum in enumerate(gate_enum):
        if s.startswith(enum):
            return idx
    raise Exception(f'{s} is an unknown gate name')

gate_type_ids = [gate_enum_to_id(x[1]) for x in cells]

connections_d = {}
for x in connections:
    if x[0].split(':')[0] == x[1].split(':')[0]:
        # same gate connection
        continue
    if x[0] not in connections_d:
        connections_d[x[0]] = set()
    connections_d[x[0]].add(x[1])

gate_d = {
    y: idx
    for idx, y in enumerate(pis + pos + [x[0] for x in cells])
}
pin_d = { 'A': 0, 'B': 1, 'Y': 2 }

def get_gate_and_pin_id(s):
    if ':' in s:
        gate_name, pin_name = s.split(':')
        return gate_d[gate_name], pin_d[pin_name]
    elif s in pis:
        return gate_d[s], 2
    elif s in pos:
        return gate_d[s], 0
    else:
        raise Exception(f'{s} is an unknown gate:pin')

# output
# print(len(gate_d))
print(len(pis))
print(len(pos))
print(len(cells))

print(len(connections_d))
for k, v in connections_d.items():
    print(
        ' '.join(map(str, get_gate_and_pin_id(k))),
        len(v),
        *map(
            lambda x: ' '.join(map(str, x)),
            map(get_gate_and_pin_id, v)
        )
    )

print(' '.join(map(str, gate_type_ids)))