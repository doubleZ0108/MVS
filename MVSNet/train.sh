#!/usr/bin/env bash
# V100
MVS_TRAINING="/media/public/yan1/doublez/realdoubleZ/Data/MVS/train/dtu"
# Lab
# MVS_TRAINING="/home/doublez/Data/MVS/train/dtu"

CUDA_VISIBLE_DEVICES=5,6,7 python train.py \
    --dataset=dtu_yao \
    --batch_size=6 \
    --trainpath=$MVS_TRAINING \
    --trainlist lists/dtu/train.txt \
    --testlist lists/dtu/test.txt \
    --numdepth=192 \
    --logdir ./checkpoints/att_c $@
