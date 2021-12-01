# Shell script for evaluating the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-29

# Dataset
# DATASET_ROOT="./dataset/dtu-test-1200/"
DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/CVP-MVSNet/dtu-test-1200/"

# Checkpoint
# LOAD_CKPT_DIR="./checkpoints/pretrained/model_000027.ckpt"
LOAD_CKPT_DIR="./checkpoints/exp/train_dtu_128/model_000000.ckpt"

# Logging
LOG_DIR="./logs/"

# Output dir
OUT_DIR="./outputs_pretrained/"

CUDA_VISIBLE_DEVICES=0 python eval.py \
\
--info="eval_pretrained_e27" \
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