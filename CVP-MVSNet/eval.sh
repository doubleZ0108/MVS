# Shell script for evaluating the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-29

# Dataset
# DATASET_ROOT="./dataset/dtu-test-1200/"
DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-test-1200/"


# @TODO
THISNAME="buf"

LOAD_CKPT_DIR="./checkpoints/"$THISNAME"/train_dtu_128/model_000022.ckpt"

OUT_DIR="/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/CVP-MVSNet/"$THISNAME

# Logging
LOG_DIR="./logs/"$THISNAME"/"

mkdir -p $LOG_DIR


CUDA_VISIBLE_DEVICES=4 python eval.py \
\
--info="test_"$THISNAME \
--mode="test" \
\
--dataset_root=$DATASET_ROOT \
--imgsize=1200 \
--nsrc=4 \
--nscale=5 \
\
--batch_size=1 \
\
--loadckpt=$LOAD_CKPT_DIR \
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
\
--outdir=$OUT_DIR