# -*- coding: utf-8 -*-
from Bio.PDB import *
from Bio.Cluster import distancematrix
import numpy as np
#import h5py
import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from PIL import Image
from Bio.PDB import PDBParser, Structure, PDBIO, Chain, Model

# 
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def label(pdb_path, pdb_name):
    parser = PDBParser()
    # 定义过滤器函数，仅保留ATOM记录
    structure = parser.get_structure(pdb_name, pdb_path)
    # 创建一个新的Structure对象
    new_structure = Structure.Structure('new_structure')
    # 创建一个集合，用于存储已添加的残基的唯一标识符
    #added_residues = set()
     # 遍历原始结构中的链
    # Iterate over each model in the structure (assumed to be dimer)
    for model in structure:
        # Iterate over each chain in the model
         # 创建一个新的Model对象，使用模型计数器的值作为标识符
        new_model = Model.Model(model.get_id())
        for chain in model:

            # 创建一个新的Chain对象
            new_chain = Chain.Chain(chain.get_id())
            for residue in chain:
                # 检查是否为氨基酸残基
                if is_aa(residue):
                    # 构建残基的唯一标识符
                    #residue_id = (chain.get_id(), residue.get_id())
                    # 检查残基是否已添加过
                    #if residue_id in added_residues:
                        #continue
                    new_chain.add(residue)
                    # 添加残基的唯一标识符到集合中
                    #added_residues.add(residue_id)
            # 将链添加到新的Structure对象中
            new_model.add(new_chain)
        new_structure.add(new_model)
    
    # 将氨基酸部分写入新的PDB文件
    #io = PDBIO()
    #io.set_structure(new_structure)
    #io.save("./new_structure.pdb")
    #structure = parser.get_structure(pdb_name, pdb_path)

    # 记录残基号
    chain_list = Selection.unfold_entities(new_structure, 'C')
    print(len(chain_list))
    if len(chain_list)<2:
        return
    res_list_A = Selection.unfold_entities(chain_list[0], 'R')
    res_list_B = Selection.unfold_entities(chain_list[1], 'R')
    #res_list_C = Selection.unfold_entities(chain_list[2], 'R')
    resseq_res_list_A = [residue.get_id()[1] for residue in res_list_A]
    resseq_res_list_B = [residue.get_id()[1] for residue in res_list_B]
    #resseq_res_list_C = [residue.get_id()[1] for residue in res_list_C]
    l_A = len(resseq_res_list_A)
    l_B = len(resseq_res_list_B)
    chains=[l_A, l_B]
    chains_str = ' '.join(str(value) for value in chains)
    filename='./contact/valid_chains/' + pdb_name + '.txt'
    #filename='./contact/train_chains/' + pdb_name + '.txt'
    # 将字符串写入文件
    with open(filename, 'w') as file:
        file.write(chains_str)
    #np.savetxt(os.path.join('./example', pdb_name + '_label.cmap'), chains, fmt='%d')
    #l_C = len(resseq_res_list_C)
    residues_length = l_A + l_B
    #residues_length = l_A + l_B +1_C

    # index，记录每个残基的原子个数
    residues = new_structure.get_residues()
    index = [0]
    for i, residue_i in enumerate(residues):
        index.append(index[i] + len(residue_i))

    # 计算距离矩阵，原子距离最小值
    atoms = new_structure.get_atoms()
    coord = [atom.get_coord() for atom in atoms]
    dist_map = distancematrix(coord)
    dist_map = [np.sqrt(i * 3) for i in dist_map]

    dismap = np.zeros(shape=(residues_length, residues_length))
    contact = np.zeros((residues_length, residues_length))
    for i in range(residues_length):
        for j in range(residues_length):       
            if i > j:
                    dis_matrix = dist_map[index[i]:index[i + 1]]
                    dis = np.array([d[index[j]:index[j + 1]] for d in dis_matrix])
                    dismap[i][j] = round(dis.min(), 3)
                    dismap[j][i] = dismap[i][j]

                    if dismap[i][j] < 8:
                        contact[i][j] = 1
                        contact[j][i] = 1
            if i==j:
                contact[i][j] = 1               
    array = np.array(contact, dtype=np.uint8) * 255

    # 创建图像对象
    image = Image.fromarray(array, 'L')
    
    # 保存图像为.jpg格式
    filename2 = os.path.join('./contact/valid_images/', pdb_name + '.jpg')
    #filename2 = os.path.join('./contact/train_images/', pdb_name + '.jpg')

    image.save(filename2)
    
    return dismap, contact


# 测试
pdb_path = './pdb_valid/1XG7.pdb'
# pdb_path_A = './example/T0805_A.pdb'
pdb_name = '1XG7'
# # pdb_path = './example/8e4r.pdb'
# # pdb_name='8e4r'
gt = label(pdb_path, pdb_name)  # label
# 1