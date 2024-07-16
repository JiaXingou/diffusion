#!/bin/bash

# 指定文本文件路径
file="../list/valid.list"

# 创建一个临时目录用于保存下载的FASTA文件
cd pdb_valid

# 逐行读取文本文件内容
while IFS= read -r line; do
    #wget "https://www.rcsb.org/pdb/entry/$pdb_id/download" -O "$pdb_id.pdb"
    wget --no-check-certificate https://files.rcsb.org/download/$line.pdb
done < "$file"
