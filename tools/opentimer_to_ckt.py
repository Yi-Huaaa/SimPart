#!/usr/bin/env python3
"""Convert an OpenTimer circuit dump into the .ckt format (see benchmark/README.md).

usage: tools/opentimer_to_ckt.py <dump.txt> <out.ckt> <bfs|dfs>

Library cells are mapped to the supported gate types (complex cells and
flip-flops become NAND/NOR gates). Loops are broken by dropping every edge that
points back to a node already visited by a BFS (or DFS) traversal from the PIs.
"""
import sys
from collections import deque

# Pin numbering per cell: output pin 0, input pins 1, 2, ...
PIN_IDS = {
    "CLKBUF": {"A": 1, "Z": 0},
    "INV": {"A": 1, "ZN": 0},
    "AND2": {"A": 1, "B": 2, "Y": 0},
    "NAND2": {"A1": 1, "A2": 2, "ZN": 0},
    "NAND3": {"A1": 1, "A2": 2, "A3": 3, "ZN": 0},
    "NAND4": {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "ZN": 0},
    "NOR2": {"A1": 1, "A2": 2, "ZN": 0},
    "NOR3": {"A1": 1, "A2": 2, "A3": 3, "ZN": 0},
    "NOR4": {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "ZN": 0},
    "AOI21": {"A": 1, "B1": 2, "B2": 3, "ZN": 0},
    "AOI22": {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "ZN": 0},
    "DFF": {"D": 1, "CK": 2, "Q": 0},
    "OAI21": {"A": 1, "B1": 2, "B2": 3, "ZN": 0},
    "OAI22": {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "ZN": 0},
    "AND2X1": {"A": 1, "B": 2, "Y": 0},
    "INVX1": {"A": 1, "Y": 0},
    "NAND3X1": {"A": 1, "B": 2, "C": 3, "Y": 0},
}

# Gate type codes of the .ckt format; cells are matched by name prefix.
GATE_TYPES = {
    "INV": 0,
    "AND": 1,
    "OR": 2,
    "XOR": 3,
    "NAND": 4,
    "NOR": 5,
    "XNOR": 6,
    "MUX": 7,
    "CLKBUF": 8,
    "AOI21": 4,  # mapped to NAND3
    "AOI22": 4,  # mapped to NAND4
    "DFF": 4,    # mapped to NAND2
    "OAI21": 5,  # mapped to NOR3
    "OAI22": 5,  # mapped to NOR4
}

OUTPUT_PIN_SUFFIXES = ["", ":Z", ":ZN", ":Q", ":Y", ":QN"]


def gate_type_id(cell_type):
    for prefix, type_id in GATE_TYPES.items():
        if cell_type.startswith(prefix):
            return type_id
    print(f"Warning: {cell_type} is an unknown gate name. Defaulting to AND (ID 1).")
    return 1


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in ("bfs", "dfs"):
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        lines = f.read().split("\n")
    mode = sys.argv[3]

    # Sections of the dump.
    start_pi = lines.index("// Start PIs")
    start_po = lines.index("// Start POs")
    start_wire = lines.index("// Start wires")
    start_cell = lines.index("// Start cells")
    start_graph = lines.index("digraph TimingGraph {")

    pis = [x.split()[1][:-1] for x in lines[start_pi + 1 : start_po - 1]]
    pos = [x.split()[1][:-1] for x in lines[start_po + 1 : start_wire - 1]]
    cells = [(x.split()[1], x.split()[0]) for x in lines[start_cell + 1 : start_graph - 2]]
    connections = [
        tuple(y.strip(' ";') for y in x.split(" -> "))
        for x in lines[start_graph + 1 : -2]
        if "->" in x
    ]

    cell_pin_key = {name: cell_type.split("_")[0] for name, cell_type in cells}
    gate_type_ids = [gate_type_id(cell_type) for _, cell_type in cells]

    # 'GATE:OUT_PIN' -> {'GATE:IN_PIN', ...}, skipping arcs inside a cell.
    fanouts = {}
    for src, dst in connections:
        if src.split(":")[0] == dst.split(":")[0]:
            continue
        fanouts.setdefault(src, set()).add(dst)

    # Node ids: PIs, then POs, then cells.
    node_id = {name: idx for idx, name in enumerate(pis + pos + [name for name, _ in cells])}

    def node_and_pin(s):
        if ":" in s:
            gate, pin = s.split(":")
            key = cell_pin_key.get(gate)
            if key is None:
                print(f"Warning: {gate} not found in gate_name_to_pin_d_key. Using default pin mapping.")
                return node_id[gate], 0
            if key in PIN_IDS and pin in PIN_IDS[key]:
                return node_id[gate], PIN_IDS[key][pin]
            print(f"Warning: {pin} for {gate} not found in pin_d. Using default.")
            return node_id[gate], 0
        if s in pis:
            return node_id[s], 0
        if s in pos:
            return node_id[s], 1
        raise Exception(f"{s} is an unknown gate:pin")

    # Break loops: drop edges to nodes visited earlier in the traversal.
    frontier = deque(pis)
    visited = set()
    while frontier:
        cur = frontier.popleft() if mode == "bfs" else frontier.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if len(visited) % 100000 == 0:
            print("progress", len(visited))

        for suffix in OUTPUT_PIN_SUFFIXES:
            if cur + suffix in fanouts:
                key = cur + suffix
                break
        else:
            if cur not in pos:
                print(f"Warning: No connection found for {cur}. Skipping this node.")
            continue
        for out in [x.split(":")[0] for x in fanouts[key]]:
            if out in visited:
                fanouts[key] = {x for x in fanouts[key] if not x.startswith(out)}
            else:
                frontier.append(out)

    with open(sys.argv[2], "w") as out:
        print(len(pis), file=out)
        print(len(pos), file=out)
        print(len(cells), file=out)
        print(len(fanouts), file=out)
        for src, dsts in fanouts.items():
            print(" ".join(map(str, node_and_pin(src))), len(dsts),
                  *(" ".join(map(str, node_and_pin(d))) for d in dsts), file=out)
        print(" ".join(map(str, gate_type_ids)), file=out)


if __name__ == "__main__":
    main()
