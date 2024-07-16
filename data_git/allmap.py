# -*- coding: utf-8 -*-
from gen_map import label
import os
def traverse_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()# 去除行尾的换行符和空格
            pdb_path = os.path.join('./pdb/', line + '.pdb')
            label(pdb_path,line)  # 调用函数并传入每行的文本内容

#file_path = './list/test.txt'  # 替换为你的.txt文件的路径
#traverse_file(file_path)  # 遍历文件并调用函数
def traverse_folder(folder_path):
    with open('valid.txt', 'w') as train_file:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdb'):
                pdb_path = os.path.join(folder_path, file_name)
                file_name_out = os.path.splitext(file_name)[0]  # 去除后缀
                print(file_name_out)
                label(pdb_path, file_name_out)  # 调用函数并传递文件路径和文件名
                train_file.write(file_name_out + '\n')  # 将文件名（去除后缀）逐行写入train.txt文件

#folder_path = './pdb/'  # 替换为你的PDB文件夹路径
folder_path = './pdb_valid/'
traverse_folder(folder_path)  # 遍历文件夹并调用函数