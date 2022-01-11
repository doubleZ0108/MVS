#!/usr/bin/env bash

DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/test/dtu/origin"
# DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/test/tankandtemples/intermediate/"
DATASET_LIST="lists/dtu/test.txt"
# DATASET_LIST="lists/tanksandtemples/test.txt"

# @TODO * 2
THISNAME='baseline'
CKPTNUM="10"

CKPT_FILE="./checkpoints/"$THISNAME"/model_0000"$CKPTNUM".ckpt"

OUTPUTNAME=$THISNAME"_"$CKPTNUM
# OUT_FILE="/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/MVSNet/"$OUTPUTNAME
OUT_FILE="mvstest/"

CUDA_VISIBLE_DEVICES=7 python eval.py \
    --dataset=dtu_yao_eval \
    --batch_size=1 \
    --testpath=$DATASET_ROOT \
    --testlist $DATASET_LIST \
    --outdir $OUT_FILE \
    --loadckpt $CKPT_FILE $@
