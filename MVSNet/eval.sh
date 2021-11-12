#!/usr/bin/env bash
DTU_TESTING="/home/doublez/Data/MVS/test/dtu/"
CKPT_FILE="./checkpoints/d192/model_000014.ckpt"
python eval.py \
    --dataset=dtu_yao_eval \
    --batch_size=1 \
    --testpath=$DTU_TESTING \
    --testlist lists/dtu/test.txt \
    --loadckpt $CKPT_FILE $@
