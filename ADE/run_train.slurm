#!/bin/bash
#SBATCH -J train                  # 作业名为 train
#SBATCH -p short                  # 作业提交的分区为 short
#SBATCH -o slurm_log/baseline/base_train.out  # 输出重定向
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH -t 20:00:00               # 任务运行的最长时间为 20 小时
#SBATCH --gres=gpu:4              # 单个节点使用 4 块 GPU 卡
#SBATCh -w espresso-0-14          # 指定运行作业的节点是 espresso-0-14，若不填写系统自动分配节点

export DETECTRON2_DATASETS=/datasets/
python train.py --num-gpus 4 --config-file configs/base_train.yaml
