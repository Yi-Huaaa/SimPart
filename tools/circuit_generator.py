#!/usr/bin/env python3
"""Generate a random combinational circuit in the .ckt format (printed to stdout).

usage: tools/circuit_generator.py [--pis 206] [--pos 107] [--gates 162000] [--seed S] > out.ckt

Gates pick their inputs among the PIs and the gates created before them; dangling
outputs are then merged by two-input gates until exactly #POs signals remain,
which drive the POs. The defaults give a circuit of the size of b19.
"""
import argparse
import random
from collections import defaultdict

PI, PO = -2, -1
NUM_INPUTS = {PI: 0, PO: 1, 0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}  # INV, AND, ..., XNOR
MAX_TYPE = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pis", type=int, default=206)
    parser.add_argument("--pos", type=int, default=107)
    parser.add_argument("--gates", type=int, default=162000, help="gates added before merging")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    random.seed(args.seed)

    gates = {i: (PI, []) for i in range(args.pis)}
    gates.update({args.pis + i: (PO, []) for i in range(args.pos)})

    # Grow the circuit.
    candidates = list(range(args.pis))
    for _ in range(args.gates):
        t = random.randint(0, MAX_TYPE)
        g = len(gates)
        gates[g] = (t, random.sample(candidates, NUM_INPUTS[t]))
        candidates.append(g)

    fanouts = defaultdict(list)
    for g, (_, inputs) in gates.items():
        for idx, x in enumerate(inputs):
            fanouts[x].append((g, idx))

    # Merge unused signals until one remains per PO.
    unused = [i for i in list(range(args.pis)) + list(range(args.pis + args.pos, len(gates)))
              if not fanouts[i]]
    while len(unused) > args.pos:
        t = random.randint(1, MAX_TYPE)
        g = len(gates)
        a, b = random.sample(unused, 2)
        gates[g] = (t, [a, b])
        unused.remove(a)
        unused.remove(b)
        unused.append(g)
    assert len(unused) == args.pos, "not enough unused signals for the POs"
    for i in range(args.pos):
        gates[args.pis + i] = (0, [unused[i]])

    fanouts = defaultdict(list)
    for g, (_, inputs) in gates.items():
        for idx, x in enumerate(inputs):
            fanouts[x].append((g, idx + 1))  # pin 0 is the output

    num_gates = len(gates) - args.pis - args.pos
    print(args.pis)
    print(args.pos)
    print(num_gates)
    print(len(gates) - args.pos)
    for g in range(len(gates)):
        if fanouts[g]:
            sinks = " ".join(f"{x[0]} {x[1]}" for x in fanouts[g])
            print(f"{g} 0 {len(fanouts[g])} {sinks}")
    print(" ".join(str(gates[g][0]) for g in range(args.pis + args.pos, len(gates))))


if __name__ == "__main__":
    main()
