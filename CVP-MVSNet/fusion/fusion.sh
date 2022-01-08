# Shell script for running fusibile depth fusion
# by: Jiayu Yang
# date: 2019-11-05

# DTU_TEST_ROOT="../dataset/dtu-test-1200"
DTU_TEST_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-test-1200/"
OUT_FOLDER="fusibile_fused"
FUSIBILE_EXE_PATH="./fusibile"

# @TODO
THISNAME="baseline_12"

DEPTH_FOLDER="/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/CVP-MVSNet/"$THISNAME

python2 depthfusion.py \
    --dtu_test_root=$DTU_TEST_ROOT \
    --depth_folder=$DEPTH_FOLDER \
    --out_folder=$OUT_FOLDER \
    --fusibile_exe_path=$FUSIBILE_EXE_PATH \
    --prob_threshold=0.8 \
    --disp_threshold=0.13 \
    --num_consistent=3
