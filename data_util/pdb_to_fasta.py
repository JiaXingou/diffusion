import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB import PDBParser

def pdb_to_fasta(input_pdb_file):
    # Create a PDB parser
    parser = PDBParser(QUIET=True)

    # Parse the PDB file
    structure = parser.get_structure('dimer', input_pdb_file)

    # Initialize an empty list to store FASTA sequences for each chain
    fasta_sequences = []

    # Iterate over each model in the structure (assumed to be dimer)
    for model in structure:
        # Iterate over each chain in the model
        for chain in model:
            # Get the chain ID
            chain_id = chain.get_id()

            # Get the chain sequence and remove any non-standard residues (e.g., HETATM)
            residues = [residue for residue in chain if residue.id[0] == ' ']

            # Convert the list of residues to a single-letter amino acid sequence
            sequence = ''.join(res.get_resname() for res in residues)

            # Create a Biopython SeqRecord object
            record = SeqIO.SeqRecord(Seq(sequence), id=chain_id, description="")

            # Append the SeqRecord to the list
            fasta_sequences.append(record)

    return fasta_sequences

def write_fasta_files(pdb_dir, pdb_name, fasta_sequences):
    # Write each chain's sequence to a separate FASTA file
    for seq_record in fasta_sequences:
        chain_id = seq_record.id
        fasta_file = os.path.join(pdb_dir, f"{pdb_name}_{chain_id}.fasta")
        SeqIO.write(seq_record, fasta_file, "fasta")
        print(f"FASTA file for Chain {chain_id} has been written to {fasta_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python <script_name.py> <input_pdb_file>")
        sys.exit(1)

    input_pdb_file = sys.argv[1]

    pdb_dir, filename = os.path.split(input_pdb_file)
    pdb_name, _ = os.path.splitext(filename)

    # Get FASTA sequences for each chain in the PDB file
    fasta_sequences = pdb_to_fasta(input_pdb_file)

    # Write each chain's FASTA sequence to a separate file
    write_fasta_files(pdb_dir, pdb_name, fasta_sequences)
