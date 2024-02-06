import argparse
import random
import math as m

# generate faults 
# generate faults, 1 wire can generate SA0 and SA1 faults 
def generate_fault_patterns(num_of_PIs_gates_POs, num_of_faults, num_of_PIs, num_of_POs):
    if num_of_faults > (2*(num_of_PIs_gates_POs-num_of_POs)):
        raise ValueError("num_of_faults must be less than or equal to (2*(num_of_PIs_gates_POs-num_of_POs)).")

    fault_patterns = set()

    while len(fault_patterns) < num_of_faults:
        wire_index = random.randint(0, num_of_PIs_gates_POs - 1)
        # print("hello1", wire_index, num_of_PIs, num_of_POs)
        while((wire_index >= num_of_PIs) and (wire_index < (num_of_PIs+num_of_POs))): # it is a PO gate
            wire_index = random.randint(0, num_of_PIs_gates_POs - 1)
            # print("hello3", wire_index, num_of_PIs, num_of_POs)

        # print("hello2", wire_index, num_of_PIs, num_of_POs)
        fault_value = random.choice([0, 1])
        fault_patterns.add((wire_index, fault_value))

    return fault_patterns

def write_fault_patterns_to_file(num_of_faults, fault_patterns, output_file):
    with open(output_file, 'w') as file:
        file.write(str(num_of_faults) + '\n')
        for wire_index, fault_value in fault_patterns:
            file.write(f"{wire_index} {fault_value}\n")


# generate patterns 
def generate_patterns(num_of_PIs, num_of_output_patterns):
    if num_of_PIs <= 0 or num_of_output_patterns <= 0:
        raise ValueError("Both num_of_PIs and num_of_output_patterns should be positive integers.")

    if pow(2, num_of_PIs) < num_of_output_patterns:
        raise Exception("num_of_output_patterns is too large.")
    
    # When add new thing into set, if the data is deplicated, it won't be added into set.
    patterns = set()
    
    while len(patterns) < num_of_output_patterns:
        new_pattern = tuple(random.choice([0, 1]) for _ in range(num_of_PIs))
        patterns.add(new_pattern)
    
    return patterns

def write_patterns_to_file(num_of_patterns, patterns, output_file):
    with open(output_file, 'w') as file:
        file.write(str(num_of_patterns) + '\n')
        for pattern in patterns:
            file.write("".join(map(str, pattern)) + '\n')

def write_patterns_to_file_new(num_of_PIs, num_of_patterns, patterns, output_file, test_bits_per_round):
    with open(output_file, 'w') as file:
        file.write(f"{num_of_patterns}\n")

        # Iterate over each bit position
        bits_per_round = int(test_bits_per_round)
        team = 1
        if num_of_patterns > bits_per_round:
            team = int(m.ceil(num_of_patterns/bits_per_round))
        
        # print decimal format
        for t in range(team):
            for b in range(num_of_PIs):
                # Extract the bits for the current bit position from all patterns
                bits_for_position = [str(pattern[b]) for pattern in patterns]
                
                accum_bits = t*bits_per_round
                # Break the continuous string of bits into chunks of `bits_per_round` bits
                chunks_of_X_bits = ["".join(bits_for_position[accum_bits:accum_bits+bits_per_round])]
                
                # Write the chunks for the current bit position
                file.write(str(int(" ".join(chunks_of_X_bits), 2)) + ' ')
            file.write('\n')
        file.write('\n')

        # print binary format -> used to check wehther the decimal format is corrected
        for t in range(team):
            for b in range(num_of_PIs):
                # Extract the bits for the current bit position from all patterns
                bits_for_position = [str(pattern[b]) for pattern in patterns]
                
                accum_bits = t*bits_per_round
                # Break the continuous string of bits into chunks of `bits_per_round` bits
                chunks_of_X_bits = ["".join(bits_for_position[accum_bits:accum_bits+bits_per_round])]
                
                # Write the chunks for the current bit position
                file.write(" ".join(chunks_of_X_bits) + ' ')
            file.write('\n')
        file.write('\n')


def write_simple_testcase_to_file(output_flst_file, output_ptn_file):
    with open(output_flst_file, 'w') as file:
        file.write("2" + '\n')
        file.write("2 1" + '\n')
        file.write("4 0" + '\n')
        file.write('\n')
        file.write('The order follows ot\'s results:' + '\n')
        file.write('File structure:' + '\n')
        file.write("2: num_of_faults" + '\n')
        file.write("2 1: PI_A: SA_1" + '\n')
        file.write("4 0: G3: SA_0" + '\n')
        file.write('\n')

    with open(output_ptn_file, 'w') as file:
        file.write("3" + '\n')
        file.write("2" + '\n')
        file.write("4" + '\n')
        file.write("1" + '\n')
        file.write('\n')
        file.write('The order follows ot\'s results:' + '\n')
        file.write('File structure:' + '\n')
        file.write("3 3: num_of_PIs, num_pattern" + '\n')
        file.write("010: PI_0 (C)'s inputs, here binary" + '\n')
        file.write("100: PI_1 (B)'s inputs, here binary" + '\n')
        file.write("001: PI_2 (A)'s inputs, here binary" + '\n')
        file.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Generate unique fault patterns and write them to a file.')
    parser.add_argument('simple', type=int, help='Generate simple testcase')
    parser.add_argument('num_of_PIs_gates_POs', type=int, help='Number of PIs + gates + POs')
    parser.add_argument('num_of_faults', type=int, help='Number of fault patterns to generate')
    parser.add_argument('output_flst_file', type=str, help='Output file to write the fault patterns')
    parser.add_argument('num_of_PIs', type=int, help='Number of PIs')
    parser.add_argument('num_of_POs', type=int, help='Number of POs')
    parser.add_argument('num_of_output_patterns', type=int, help='Number of output patterns to generate')
    parser.add_argument('output_ptn_file', type=str, help='Output file to write the patterns')
    parser.add_argument('test_bits_per_round', type=str, help='test_bits_per_round, fo example: 32 (uint), 64 (size_t in 64 bits)')
    args = parser.parse_args()

    if (args.simple == 1):
        write_simple_testcase_to_file(args.output_flst_file, args.output_ptn_file)
    else:
        print("Generate faults file: args.output_flst_file")
        fault_patterns = generate_fault_patterns(args.num_of_PIs_gates_POs, args.num_of_faults, args.num_of_PIs, args.num_of_POs)
        write_fault_patterns_to_file(args.num_of_faults, fault_patterns, args.output_flst_file)

        print("Generate patterns file:", args.output_ptn_file)
        generated_patterns = generate_patterns(args.num_of_PIs, args.num_of_output_patterns)
        write_patterns_to_file_new(args.num_of_PIs, args.num_of_output_patterns, generated_patterns, args.output_ptn_file, args.test_bits_per_round)



if __name__ == "__main__":
    main()
