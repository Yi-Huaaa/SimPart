#!/usr/bin/env python3
"""Build a larger benchmark by placing N independent copies of a circuit side by side.

This is how the "_eval" benchmarks of the paper were generated (8 or 16 copies).
Node indices of the result keep the .ckt convention: all PIs first (copy by
copy), then all POs, then all gates. Every copy receives the same input patterns.

usage: tools/scale_benchmark.py N in.ckt in.ptn out.ckt out.ptn
"""
import argparse


def scale_circuit(src, dst, copies):
    with open(src) as f:
        lines = f.readlines()
    # Line layout: #PIs, #POs, #gates, #wires, one line per wire, gate types.
    num_pis, num_pos, num_gates, num_wires = (int(lines[i].split()[0]) for i in range(4))
    wires = lines[4:-1]
    gate_types = lines[-1].strip()
    assert len(wires) == num_wires, "expected one wire per line"

    def shifter(copy):
        pi_offset = copy * num_pis
        po_offset = (copies - 1) * num_pis + copy * num_pos
        gate_offset = (copies - 1) * (num_pis + num_pos) + copy * num_gates

        def shift(v):
            if v < num_pis:
                return v + pi_offset
            if v < num_pis + num_pos:
                return v + po_offset
            return v + gate_offset
        return shift

    with open(dst, "w") as out:
        for value in (num_pis, num_pos, num_gates, num_wires):
            out.write(f"{value * copies}\n")
        for copy in range(copies):
            shift = shifter(copy)
            for wire in wires:
                # <driver> <pin> <#sinks> (<sink> <pin>)*: shift the driver and the sinks.
                fields = list(map(int, wire.split()))
                fields[0] = shift(fields[0])
                for i in range(3, len(fields), 2):
                    fields[i] = shift(fields[i])
                out.write(" ".join(map(str, fields)) + "\n")
        out.write(" ".join([gate_types] * copies) + "\n")


def scale_patterns(src, dst, copies):
    with open(src) as f:
        lines = f.readlines()
    with open(dst, "w") as out:
        out.write(lines[0])  # number of patterns
        for line in lines[1:]:  # one line (round) of PI words, repeated for every copy
            out.write((line.strip() + " ") * copies + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("copies", type=int)
    parser.add_argument("circuit")
    parser.add_argument("patterns")
    parser.add_argument("out_circuit")
    parser.add_argument("out_patterns")
    args = parser.parse_args()
    scale_circuit(args.circuit, args.out_circuit, args.copies)
    scale_patterns(args.patterns, args.out_patterns, args.copies)


if __name__ == "__main__":
    main()
