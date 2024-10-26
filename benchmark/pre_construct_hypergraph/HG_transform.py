import os

def process_data(input_file, output_file):
    # Dictionary to store different sections
    data_sections = {
        '_hyperedge_include_which_vertices': [],
        'vertex_include_which_hyperedges': [],
        '_hyperedges_weight': [],
        '_hyperedges_eptr': [],
        '_hyperedges_eind': [],
        '_vectices_weight': [],
        '_vertexIdx_to_sinkGateIdx': [],
        '_sinkGateIdx_to_vertexIdx': []            
    }

    # Reading the input file
    with open(input_file, 'r') as file:
        current_section = None
        for line in file:
            line = line.strip()
            # Skip comments or empty lines
            if not line or line.startswith('//'):
                continue
            
            # Identify sections
            if line.startswith('_hyperedge_include_which_vertices:'):
                current_section = '_hyperedge_include_which_vertices'
                continue
            elif line.startswith('vertex_include_which_hyperedges:'):
                current_section = 'vertex_include_which_hyperedges'
                continue
            elif line.startswith('_hyperedges_weight:'):
                current_section = '_hyperedges_weight'
                continue
            elif line.startswith('_hyperedges_eptr:'):
                current_section = '_hyperedges_eptr'
                continue
            elif line.startswith('_hyperedges_eind:'):
                current_section = '_hyperedges_eind'
                continue
            elif line.startswith('_vectices_weight'):
                current_section = '_vectices_weight'
                continue
            elif line.startswith('_vertexIdx_to_sinkGateIdx'):
                current_section = '_vertexIdx_to_sinkGateIdx'
                continue
            elif line.startswith('_sinkGateIdx_to_vertexIdx'):
                current_section = '_sinkGateIdx_to_vertexIdx'
                continue                        

            # Process line under the current section
            if current_section:
                # Remove index column from sections that include it
                if current_section in ['_hyperedge_include_which_vertices', 'vertex_include_which_hyperedges', '_hyperedges_weight', '_vectices_weight', '_vertexIdx_to_sinkGateIdx', '_sinkGateIdx_to_vertexIdx']:
                    parts = line.split()
                    data_sections[current_section].append(parts[1:])
                else:
                    # For _hyperedges_eptr and _hyperedges_eind, add line as-is
                    data_sections[current_section].append(line)

    # Writing to the output file
    with open(output_file, 'w') as file:
        file.write("1\n")
        # Process and write each section according to required format

        # _hyperedge_include_which_vertices section
        file.write(f"{len(data_sections['_hyperedge_include_which_vertices'])}\n")
        for item in data_sections['_hyperedge_include_which_vertices']:
            file.write(f"{len(item)} {' '.join(item)}\n")
        file.write("\n")

        # vertex_include_which_hyperedges section
        file.write(f"{len(data_sections['vertex_include_which_hyperedges'])}\n")
        for item in data_sections['vertex_include_which_hyperedges']:
            file.write(f"{len(item)} {' '.join(item)}\n")
        file.write("\n")

        # _hyperedges_weight section
        file.write(f"{len(data_sections['_hyperedges_weight'])}\n")
        weights = ' '.join([item[0] for item in data_sections['_hyperedges_weight']])
        file.write(f"{weights}\n\n")

        # _hyperedges_eptr section
        file.write(f"{len(data_sections['_hyperedges_eptr'])}\n")
        file.write(" ".join(data_sections['_hyperedges_eptr']) + "\n\n")

        # _hyperedges_eind section
        file.write(f"{len(data_sections['_hyperedges_eind'])}\n")
        file.write(" ".join(data_sections['_hyperedges_eind']) + "\n\n")

        # _vectices_weight section
        file.write(f"{len(data_sections['_vectices_weight'])}\n")
        vertices_weights = ' '.join([item[0] for item in data_sections['_vectices_weight']])
        file.write(f"{vertices_weights}\n\n")

        # _vertexIdx_to_sinkGateIdx section
        file.write(f"{len(data_sections['_vertexIdx_to_sinkGateIdx'])}\n")
        vertices_weights = ' '.join([item[0] for item in data_sections['_vertexIdx_to_sinkGateIdx']])
        file.write(f"{vertices_weights}\n\n")

        # _sinkGateIdx_to_vertexIdx section
        file.write(f"{len(data_sections['_sinkGateIdx_to_vertexIdx'])}\n")
        vertices_weights = ' '.join([item[0] for item in data_sections['_sinkGateIdx_to_vertexIdx']])
        file.write(f"{vertices_weights}\n")                

# # Specify the input and output file paths
# input_file = 'input.txt'
# output_file = 'output_1.txt'

# # Process the data and create the output file
# process_data(input_file, output_file)


# List of input files
input_files = [
    'RepCutOriginal.txt', 
    'leon2_iccad_eval.txt', 'mgc_edit_dist_iccad.txt', 
    'b19_iccad.txt', 
    # 'leon3mp_iccad.txt', 'mgc_matrix_mult_iccad.txt', 'netcard_iccad.txt'
    # 'c17.txt', 'c432.txt', 'c499.txt', 'c880.txt', 'c1355.txt', 
    # 'c1908.txt', 'c2670.txt', 'c3540.txt', 'c5315.txt', 'c7552.txt',
]

# Process each file and create the output file
for input_file in input_files:
    # Generate output file name by adding "_output" before ".txt"
    output_file = f"./ok_inputs/{os.path.splitext(input_file)[0]}_output.txt"
    process_data(input_file, output_file)