# -*- coding: utf-8 -*-

#从单链文件中去除非标准残基

#单个文件夹
#import os

#input_folder = "6D2V"  # 输入文件夹的路径

# 遍历输入文件夹中的所有文件
#for filename in os.listdir(input_folder):
#    if filename.endswith(".pdb") and "chain" in filename.lower():
#        input_file = os.path.join(input_folder, filename)
#        output_file = os.path.join(input_folder, filename[4:])

#        with open(input_file, "r") as file:
#            lines = file.readlines()

        #filtered_lines = [line for line in lines if not line.startswith("HETATM")]
#        filtered_lines = [line for line in lines if not line.startswith("HETATM") and "HOH" not in line]
#        
#        with open(output_file, "w") as file:
#            file.writelines(filtered_lines)

#大文件里面包含所有的id
import os

#input_folder = "pdb_out"  # 大文件夹路径
#代码文件夹/文件夹/文件
#input_folder="/extendplus/zhaoqian/PGT-wt/data/3dhe/3dhe_2750/"
#input_folder="/extendplus/zhaoqian/PGT-wt/data/3dho/3dho_7997/"
#input_folder="/extendplus/zhaoqian/PGT-wt/data/DB5/DB_59/"
input_folder="/extendplus/zhaoqian/PGT-wt/data/PLMG/PL_7362/"
#input_folder="/extendplus/zhaoqian/PGT-wt/data/DHT/DHT_130/"
#input_folder="/extendplus/zhaoqian/PGT-wt/data/HeteroPDB/He_200/"

# 遍历大文件夹中的所有文件夹
for foldername in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, foldername)
    #print(folder_path)
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        # 遍历小文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdb") and "chain" in filename.lower():
                input_file = os.path.join(folder_path, filename)
                output_file = os.path.join(folder_path, filename[4:])
                #print(output_file)
                with open(input_file, "r") as file:
                    lines = file.readlines()

                filtered_lines = [line for line in lines if not line.startswith("HETATM") and "HOH" not in line]

                with open(output_file, "w") as file:
                    file.writelines(filtered_lines)