#!/usr/bin/env bash
TESTPATH="/media/public/yan1/doublez/realdoubleZ/Data/MVS/test/dtu"
TESTLIST="lists/dtu/test.txt"

CKPT_FILE="checkpoints/mytest/model_000015.ckpt"
OUTPUT_DIR="./outputs/baseline"

CUDA_VISIBLE_DEVICES=2 python test.py --dataset=general_eval \
    --batch_size=1 \
    --testpath=$TESTPATH \
    --testlist=$TESTLIST \
    --loadckpt=$CKPT_FILE \
    --outdir=$OUTPUT_DIR  \
    --interval_scale=1.06 \
    --filter_method="gipuma"
