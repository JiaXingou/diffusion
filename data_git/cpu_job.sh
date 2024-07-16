#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project改为导师创建的课题组名
#SBATCH --comment=Bioinfo_MIALAB

### 给您这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=downpdb

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL等），下面参数写1！不要多写，多写了也不会加速程序！
#SBATCH --nodes=1

### 指定该作业需要多少个CPU核心
### 注意！一般根据队列的CPU核心数填写，比如cpu队列64核，这里申请64核，并在你的程序中尽量使用多线程充分利用64核资源！
#SBATCH --ntasks=24

### 指定该作业在哪个队列上执行
#SBATCH --partition=cpu24c

### 以上参数用来申请所需资源
### 以下命令将在计算节点执行
### 以下命令将在计算节点执行

###本例使用Anaconda中的Python，先将Python添加到环境变量配置好环境变量
export PATH=/home/u2022103607/anaconda3/envs/diffusion/bin:$PATH
### 激活一个 Anaconda 环境 tf22
conda activate diffusion
#source activate alphafold

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin

### 执行你的作业

python allmap.py
