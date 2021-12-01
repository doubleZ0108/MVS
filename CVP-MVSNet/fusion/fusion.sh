# Shell script for running fusibile depth fusion
# by: Jiayu Yang
# date: 2019-11-05

# DTU_TEST_ROOT="../dataset/dtu-test-1200"
DTU_TEST_ROOT="/media/public/yan1/doublez/realdoubleZ/Data/CVP-MVSNet/dtu-test-1200/"

DEPTH_FOLDER="../outputs_pretrained/"
OUT_FOLDER="fusibile_fused"
FUSIBILE_EXE_PATH="./fusibile"

python depthfusion.py \
--dtu_test_root=$DTU_TEST_ROOT \
--depth_folder=$DEPTH_FOLDER \
--out_folder=$OUT_FOLDER \
--fusibile_exe_path=$FUSIBILE_EXE_PATH \
--prob_threshold=0.8 \
--disp_threshold=0.13 \
--num_consistent=3
