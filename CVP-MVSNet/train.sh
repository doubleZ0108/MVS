# Shell script for training the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-13

# Dataset
# DATASET_ROOT="./dataset/dtu-train-128/"
DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-train-128/"

# @TODO
# Logging
THISNAME="buf"
CKPT_DIR="./checkpoints/"$THISNAME"/"
LOG_DIR="./logs/"$THISNAME"/"

mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=5 python train.py \
\
--info="train_dtu" \
--mode="train" \
\
--dataset_root=$DATASET_ROOT \
--imgsize=128 \
--nsrc=2 \
--nscale=2 \
\
--epochs=40 \
--lr=0.001 \
--lrepochs="10,12,14,20:2" \
--batch_size=6 \
\
--loadckpt='' \
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
--resume=0 \
