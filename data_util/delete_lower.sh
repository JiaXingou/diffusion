#!/bin/sh

#按照重复名单删除重复ID的pdb和fasta文件
#只能删除小写ID的文件

read_and_delete_content() {
  # 读取文件内容
  #content=$(cat "$1")

  # 逐行读取文件内容
  while IFS= read -r line; do
    # 对每一行进行处理
    #echo "$line"

    # 删除文件夹中带有文件名前缀的文件
    prefix="$line"
    #term= $(echo "$prefix" | tr '[:lower:]' '[:upper:]')
    #term= $(printf "%s" "$prefix" | tr '[:lower:]' '[:upper:]')
    #term="$prefix^^"
    #term= $(echo "$prefix"| awk '{print toupper($0)}')
    #term= $(awk '{ print toupper($0) }' <<< "$prefix")
    #echo "$term"
    find "$2" -type f -name "${prefix}*" -exec rm {} +
  done < "$1"
}

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

#对去冗余后的要去除的蛋白质进行操作，针对文件夹下的文件目录
file_path="train_cd9_diff.txt"
#folder_path="./3dhe/3d_he_2750"
folder_path="./3dho/3d_ho_7997/"


read_and_delete_content "$file_path" "$folder_path"