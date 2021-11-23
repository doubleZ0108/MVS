#!/usr/bin/env bash
# V100
DTU_TESTING="/media/public/yan1/doublez/realdoubleZ/Data/MVS/test/dtu"
# Lab
# MVS_TRAINING="/home/doublez/Data/MVS/test/dtu"

CKPT_FILE="./checkpoints/d192/model_000014.ckpt"

CUDA_VISIBLE_DEVICES=7 python eval.py \
    --dataset=dtu_yao_eval \
    --batch_size=1 \
    --testpath=$DTU_TESTING \
    --testlist lists/dtu/test.txt \
    --loadckpt $CKPT_FILE $@
