import argparse
import random

# generate faults 
# generate faults, 1 wire only gen 1 testcase
def generate_fault_patterns_wo_same_wire (num_of_wires, num_of_faults):
    if num_of_faults > num_of_wires:
        raise ValueError("num_of_faults must be less than or equal to num_of_wires.")

    fault_patterns = set()
    wires_with_faults = set()

    while len(fault_patterns) < num_of_faults:
        wire_index = random.randint(0, num_of_wires - 1)

        # Skip wires that already have a fault
        if wire_index in wires_with_faults:
            continue

        fault_value = random.choice([0, 1])
        fault_patterns.add((wire_index, fault_value))
        wires_with_faults.add(wire_index)

    return fault_patterns

# generate faults, 1 wire can generate SA0 and SA1 faults 
def generate_fault_patterns(num_of_wires, num_of_faults):
    if num_of_faults > (2*num_of_wires):
        raise ValueError("num_of_faults must be less than or equal to (2*num_of_wires).")

    fault_patterns = set()

    while len(fault_patterns) < num_of_faults:
        wire_index = random.randint(0, num_of_wires - 1)
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

# def write_patterns_to_file_new(num_of_PIs, num_of_patterns, patterns, output_file):
#     with open(output_file, 'w') as file:
#         file.write(str(num_of_PIs) + ' ' + str(num_of_patterns) + '\n')
#         for b in range(num_of_PIs):
#             file.write(''.join(str(pattern[b]) for pattern in patterns) + '\n')
#             # file.write(' '.join(str(pattern[b]) for pattern in patterns) + '\n')
            
def write_patterns_to_file_new(num_of_PIs, num_of_patterns, patterns, output_file):
    with open(output_file, 'w') as file:
        file.write(f"{num_of_PIs} {num_of_patterns}\n")

        # # Iterate over each bit position
        # for b in range(num_of_PIs):
        #     # Extract the bits for the current bit position from all patterns
        #     bits_for_position = [str(pattern[b]) for pattern in patterns]
            
        #     # Break the continuous string of bits into chunks of 32 bits
        #     chunks_of_32_bits = ["".join(bits_for_position[i:i+32]) for i in range(0, len(bits_for_position), 32)]
            
        #     # Write the chunks for the current bit position
        #     file.write(" ".join(chunks_of_32_bits) + '\n')
            
        # file.write('\n')

        # Iterate over each bit position
        team = 1
        if num_of_patterns/32 > 1:
            team = int(num_of_patterns/32)
        
        # print decimal format
        for t in range(team):
            for b in range(num_of_PIs):
                # Extract the bits for the current bit position from all patterns
                bits_for_position = [str(pattern[b]) for pattern in patterns]
                
                accum_bits = t*32
                # Break the continuous string of bits into chunks of 32 bits
                chunks_of_32_bits = ["".join(bits_for_position[accum_bits:accum_bits+32])]
                
                # Write the chunks for the current bit position
                file.write(str(int(" ".join(chunks_of_32_bits), 2)) + ' ')
            file.write('\n')
        file.write('\n')

        # print binary format
        for t in range(team):
            for b in range(num_of_PIs):
                # Extract the bits for the current bit position from all patterns
                bits_for_position = [str(pattern[b]) for pattern in patterns]
                
                accum_bits = t*32
                # Break the continuous string of bits into chunks of 32 bits
                chunks_of_32_bits = ["".join(bits_for_position[accum_bits:accum_bits+32])]
                
                # Write the chunks for the current bit position
                file.write(" ".join(chunks_of_32_bits) + ' ')
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
        file.write("3 3" + '\n')
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
    parser.add_argument('num_of_wires', type=int, help='Number of wires')
    parser.add_argument('num_of_faults', type=int, help='Number of fault patterns to generate')
    parser.add_argument('output_flst_file', type=str, help='Output file to write the fault patterns')
    parser.add_argument('num_of_PIs', type=int, help='Number of PIs (Pattern Inputs)')
    parser.add_argument('num_of_output_patterns', type=int, help='Number of output patterns to generate')
    parser.add_argument('output_ptn_file', type=str, help='Output file to write the patterns')
    args = parser.parse_args()

    if (args.simple == 1):
        write_simple_testcase_to_file(args.output_flst_file, args.output_ptn_file)
    else:
        print("Generate faults file: benchmark/simple.flst")
        fault_patterns = generate_fault_patterns(args.num_of_wires, args.num_of_faults)
        write_fault_patterns_to_file(args.num_of_faults, fault_patterns, args.output_flst_file)

        # print("Generate patterns file: benchmark/simple.ptn")
        # generated_patterns = generate_patterns(args.num_of_PIs, args.num_of_output_patterns)
        # # write_patterns_to_file(args.num_of_output_patterns, generated_patterns, args.output_ptn_file)
        # write_patterns_to_file_new(args.num_of_PIs, args.num_of_output_patterns, generated_patterns, args.output_ptn_file)

        print("Generate patterns file:", args.output_ptn_file)
        generated_patterns = generate_patterns(args.num_of_PIs, args.num_of_output_patterns)
        write_patterns_to_file_new(args.num_of_PIs, args.num_of_output_patterns, generated_patterns, args.output_ptn_file)



if __name__ == "__main__":
    main()
