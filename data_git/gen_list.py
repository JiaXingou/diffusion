# -*- coding: utf-8 -*-
import os

folder_path = './pdb'  # 指定文件夹路径
#folder_path = './contact/valid_images'
#output_file = 'valid_images.txt'  # 输出文件名
output_file = 'train.txt' 
with open(output_file, 'w') as train_file:
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.pdb'):
                file_name = file_name[:-4]  # 去掉末尾的 ".jpg" 后缀
            train_file.write(file_name + '\n')