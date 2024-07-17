#!/bin/bash

#整体运行split.py代码

#代码文件夹/文件夹/文件
#folder_path="pdb_out"  # 文件夹路径
#folder_path="/extendplus/zhaoqian/PGT-wt/data/3dhe/3dhe_2750/"
#folder_path="/extendplus/zhaoqian/PGT-wt/data/3dho/3dho_7997/"
#folder_path="/extendplus/zhaoqian/PGT-wt/data/DB5/DB_59/"
#folder_path="/extendplus/zhaoqian/PGT-wt/data/PLMG/PL_7362/"
#folder_path="/extendplus/zhaoqian/PGT-wt/data/DHT/DHT_130/"
#folder_path="/extendplus/zhaoqian/PGT-wt/data/HeteroPDB/He_200/"
folder_path="/extendplus/zhaoqian/PGT-wt/data/DeepHomo_testset/DeepHomo_testset/test_300/"

# 递归函数处理文件夹及其子文件夹
process_folder() {
    local dir=$1

    # 遍历目录中的文件和文件夹
    for item in "$dir"/*; do
        if [[ -f "$item" && $item == *.pdb ]]; then
            # 处理 PDB 文件
            file_path="$item"  # 文件的完整路径
            #file_name=$(basename "$item")  # 文件名
            output_dir=$(dirname "$item")  # 输出目录为当前文件所在的目录
            output_prefix=$(basename "${item%.*}")  # 使用文件名作为输出前缀
            #output_folder="$output_dir"  # 输出文件夹的完整路径
            command="python split.py \"$file_path\" "
            eval $command
            #echo "Processed file: $item"
        elif [[ -d "$item" ]]; then
            # 递归处理子文件夹
            process_folder "$item"
        fi
    done
}

# 调用递归函数处理顶层文件夹
process_folder "$folder_path"