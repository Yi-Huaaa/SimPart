import random
from collections import defaultdict

# configs
PI_nums = 206
PO_nums = 107
rounds = 162000

# as large as b17:
# PI_nums = 206
# PO_nums = 107
# rounds = 23000

# as large as b18:
# PI_nums = 206
# PO_nums = 107
# rounds = 85000

# as large as b19:
# PI_nums = 206
# PO_nums = 107
# rounds = 162000

#   INV = 0,
#   AND, // 1
#   OR, // 2
#   XOR,  // 3
#   NAND, // 4
#   NOR, // 5
#   XNOR, // 6
type_to_input_nums = {
    -2: 0,  # PI
    -1: 1,  # PO
    0: 1,  # INV
    1: 2,  # AND
    2: 2,  # OR
    3: 2,  # XOR
    4: 2,  # NAND
    5: 2,  # NOR
    6: 2,  # XNOR
}
type_max = len(type_to_input_nums) - 3

# states
gates = {}

# add PIs
for i in range(PI_nums):
    gates[i] = (-2, [])

# add POs
for i in range(PO_nums):
    gates[i + PI_nums] = (-1, [])

# try to expand the circuit by adding gates
candi = list(range(PI_nums))
for _ in range(rounds):
    t = random.randint(0, type_max)
    next_gate = len(gates)
    gates[next_gate] = (t, random.sample(candi, type_to_input_nums[t]))
    candi.append(next_gate)

# get the connections
results = defaultdict(list)
for k, v in gates.items():
    for idx, x in enumerate(v[1]):
        results[x].append((k, idx))

# those whose output is unused
candi = list(range(PI_nums)) + list(range(PI_nums + PO_nums, len(gates)))
candi = [i for i in candi if not results[i]]
# print(candi)

# fill if candi is not enough
# candi += random.sample([i for i in candi if results[i]], PO_nums - len(candi))

# merge the circuit
while len(candi) > PO_nums:
    t = random.randint(1, type_max)
    next_gate = len(gates)
    x1, x2 = random.sample(candi, 2)
    gates[next_gate] = (t, [x1, x2])
    candi.remove(x1)
    candi.remove(x2)
    candi.append(next_gate)
# print(candi)

# print(len(candi), PO_nums)
# exit(1)
assert len(candi) == PO_nums
for i in range(PO_nums):
    t = 0  # INV
    gates[PI_nums + i] = (t, [candi[i]])

# reassign
results = defaultdict(list)
for k, v in gates.items():
    for idx, x in enumerate(v[1]):
        results[x].append((k, idx + 1))  # + 1 since pin 0 is the output

# print PI, PO
print(PI_nums)
print(PO_nums)
print(len(gates) - PI_nums - PO_nums)
print(len(gates) - PO_nums)

# print connections
for idx in range(len(gates)):
    if results[idx]:
        s = " ".join(f"{x[0]} {x[1]}" for x in results[idx])
        print(f"{idx} 0 {len(results[idx])} {s}")

print(" ".join(str(gates[idx][0]) for idx in range(PI_nums + PO_nums, len(gates))))
