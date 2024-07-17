# -*- coding: utf-8 -*-
import os

def read_and_delete_content(filename, folder_path):
    # 读取文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 逐行读取文件内容
    for line in lines:
        # 删除换行符
        line = line.strip()

        # 删除文件夹中带有文件名前缀的文件
        prefix = line
        term = prefix.upper()
        #print(term)

        # 查找并删除匹配的文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith(term):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

# 示例用法

#对去重复后的要去除的蛋白质进行操作，针对文件夹下的文件目录
#file_path="re_he.txt"
#folder_path="./re/"
#file_path="3dhe_merged_file.txt"
#folder_path="./3dhe/3d_he_2750/"
#file_path="3dhe_merged_file.txt"
#folder_path="./3dhe/3d_he_fasta/"
#file_path="3dho_merged_file.txt"
#folder_path="./3dho/3d_ho_7997/"
#file_path="3dho_merged_file.txt"
#folder_path="./3dho/3d_ho_fasta/"
#file_path="PL_merged_file.txt"
#folder_path="./PLMG/PLMG_7362"
#file_path="PL_merged_file.txt"
#folder_path="./PLMG/PLMG_fasta"

#对去冗余后的要去除的蛋白质进行操作，针对文件夹下的文件目录
file_path="train_cd9_diff.txt"
#folder_path="./PLMG/PLMG_7362"
#folder_path="./PLMG/PLMG_fasta"
folder_path="./3dhe/3d_he_fasta"
#folder_path="./3dho/3d_ho_fasta"


read_and_delete_content(file_path, folder_path)