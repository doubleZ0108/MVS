#!/usr/bin/env bash
source /media/public/yan1/doublez/anaconda3/etc/profile.d/conda.sh
conda activate casmvsnet

MVS_TRAINING="/media/public/yan1/doublez/realdoubleZ/Data/MVS/train/dtu"

NGPUS=1

LOG_DIR="./checkpoints/debug"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS train.py \
    --logdir $LOG_DIR \
    --dataset=dtu_yao \
    --batch_size=3 \
    --trainpath=$MVS_TRAINING \
    --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt \
    --numdepth=192 \
    --eval_freq=4 | tee -a $LOG_DIR/log.txt