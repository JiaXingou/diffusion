# -*- coding: utf-8 -*-

#将pdb文件拆分成单链

import sys
from Bio.PDB import PDBParser, PDBIO
import os
import ipdb

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py input_pdb_file ")
    sys.exit(1)

# Get the input PDB file and output directory from command-line arguments
input_pdb_file = sys.argv[1]
#output_directory = sys.argv[2]

# Extract the input file name prefix
input_file_prefix = os.path.splitext(input_pdb_file)[0]
prefix = input_file_prefix[-4:]
#print(prefix)
# Create a PDBParser object
parser = PDBParser(QUIET=True)

# Load the PDB structure
structure = parser.get_structure("structure", input_pdb_file)

# Get the first model
model = structure[0]

# Get the input file's directory
input_directory = os.path.dirname(input_pdb_file)
#print(input_directory)
# Create the output directory if it doesn't exist
#output_directory = os.path.join(input_directory, "pdb_out")  # 输出目录路径
#if not os.path.exists(output_directory):
#    os.makedirs(output_directory)


# Iterate through chains and generate separate PDB files for each chain
for chain in model:
    chain_id = chain.id
    output_pdb_file = os.path.join(input_directory, "old_{}_chain_{}.pdb".format(prefix, chain_id))
    #ipdb.set_trace()
    io = PDBIO()
    io.set_structure(chain)
    io.save(output_pdb_file)

    #print("Chain {} saved to {}".format(chain_id, output_pdb_file))