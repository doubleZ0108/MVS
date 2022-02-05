# Shell script for running fusibile depth fusion


# MODIFY THE PATHS HERE
DTU_TEST_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/MVS/CVP-MVSNet/dtu-test-1200/"
DEPTH_FOLDER="../outputs/baseline_39"
OUT_FOLDER="fusibile_fused"
FUSIBILE_EXE_PATH="./fusibile"

CUDA_VISIBLE_DEVICES=5 python2 depthfusion.py \
--dtu_test_root=$DTU_TEST_ROOT \
--depth_folder=$DEPTH_FOLDER \
--out_folder=$OUT_FOLDER \
--fusibile_exe_path=$FUSIBILE_EXE_PATH \
--prob_threshold=0.6 \
--disp_threshold=0.25 \
--num_consistent=3