import os
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBIO
from Bio.PDB import Structure
def process_pdb_file(file_path, output_path):
    # 创建PDB解析器
    parser = PDBParser(QUIET=True)
    
    # 解析PDB文件
    structure = parser.get_structure('pdb', file_path)
    
    # 创建一个新的Structure对象
    new_structure = Structure.Structure('new_structure')
    # 创建一个集合，用于存储已添加的残基的唯一标识符
    added_residues = set()
    # 遍历模型、链和残基
    for model in structure:
        for chain in model:
            for residue in chain:
                # 检查是否为氨基酸残基
                if is_aa(residue):
                    # 构建残基的唯一标识符
                    residue_id = (chain.get_id(), residue.get_id())
                    
                    # 检查残基是否已添加过
                    if residue_id in added_residues:
                        continue
                    new_structure.add(residue)
                    # 添加残基的唯一标识符到集合中
                    added_residues.add(residue_id)
    
    # 将氨基酸部分写入新的PDB文件
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)

def process_pdb_files_in_folder(folder_path, output_folder):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件扩展名是否为.pdb
        if file_name.endswith('.pdb'):
            # 构建输入文件的完整路径
            file_path = os.path.join(folder_path, file_name)
            
            # 构建输出文件的完整路径
            output_file_name = f'processed_{file_name}'
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # 处理单个PDB文件
            process_pdb_file(file_path, output_file_path)
            
input_folder = './pdb_valid'  # 指定输入文件夹的路径
output_folder = './clpdb'  # 指定输出文件夹的路径

process_pdb_files_in_folder(input_folder, output_folder)