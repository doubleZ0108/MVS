# Shell script for evaluating the AACVP-MVSNet
# Modified by A.Yu

# Dataset
DATASET_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-test-1200/"

# Checkpoint
LOAD_CKPT_DIR="./checkpoints/baseline/model_000039.ckpt"
# Logging
LOG_DIR="./logs/"

# Output dir. You may want to change it here.
OUT_DIR="/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/AACVP-MVSNet/xxx"

CUDA_VISIBLE_DEVICES=5 python eval_AACVPMVSNet.py --mode="test" --dataset_root=$DATASET_ROOT \
--imgsize=1200 \
--nsrc=3 \
--nscale=5 \
--batch_size=1 \
--loadckpt=$LOAD_CKPT_DIR \
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
--outdir=$OUT_DIR
