#!/usr/bin/env bash
# V100
MVS_TRAINING="/media/public/yan1/doublez/realdoubleZ/Data/MVS/train/dtu"
# Lab
# MVS_TRAINING="/home/doublez/Data/MVS/train/dtu"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py \
    --dataset=dtu_yao \
    --batch_size=10 \
    --trainpath=$MVS_TRAINING \
    --trainlist lists/dtu/train.txt \
    --testlist lists/dtu/test.txt \
    --numdepth=192 \
    --resume \
    --logdir ./checkpoints/test1122 $@
