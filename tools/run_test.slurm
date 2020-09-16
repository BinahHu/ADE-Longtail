#!/bin/bash
#SBATCH -J train                  # 作业名为 train
#SBATCH -p short                  # 作业提交的分区为 short
#SBATCH -o test_log.out           # 输出重定向
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH -t 1:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:4              # 单个节点使用 4 块 GPU 卡
#SBATCh -w espresso-0-14          # 指定运行作业的节点是 espresso-0-14，若不填写系统自动分配节点


export DETECTRON2_DATASETS=/datasets/
python train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS output/model_0054999.pth
