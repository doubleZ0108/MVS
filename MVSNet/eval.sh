#!/usr/bin/env bash
# V100
DTU_TESTING="/media/public/yan1/doublez/realdoubleZ/Data/MVS/test/dtu"
# Lab
# MVS_TRAINING="/home/doublez/Data/MVS/test/dtu"

THISNAME='thisname'

CKPT_FILE="./checkpoints/baseline/model_000015.ckpt"

OUT_FILE="/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/MVSNet/"$THISNAME

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --dataset=dtu_yao_eval \
    --batch_size=1 \
    --testpath=$DTU_TESTING \
    --testlist lists/dtu/mytest.txt \
    --outdir $OUT_FILE \
    --loadckpt $CKPT_FILE $@
