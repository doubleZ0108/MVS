#!/usr/bin/env python
# Shell script for training the AACVP-MVSNet
# Modified by: B. Liu
# date: 2020-08-13
# Dataset
DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-train-128/"

CUDA_VISIBLE_DEVICES=1,2 python train_AACVPMVSNet.py --info="buf" --mode="train" --groups=4 --num_heads=1 \
--dataset_root=$DATASET_ROOT \
--imgsize=128 \
--nsrc=5 \
--nscale=2 \
--batch_size=14 \
--loadckpt='' \
--resume=0 \
--cuda_ids="1,2"
/