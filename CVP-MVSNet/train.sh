# Shell script for training the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-13

# Dataset
# DATASET_ROOT="./dataset/dtu-train-128/"
DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-train-128/"

# Logging
CKPT_DIR="./checkpoints/att3d-1209/"
LOG_DIR="./logs/"

CUDA_VISIBLE_DEVICES=0,6 python train.py \
\
--info="train_dtu_128" \
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
--batch_size=16 \
\
--loadckpt='' \
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
--resume=0 \
