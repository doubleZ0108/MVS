PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/MVSNet/outputs/mvsnet_results/'
RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/Evaluation/results/MVSNet/mvsnettest/'
METHOD='mvsnet'
OTHERMSG=''

# PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/CVP-MVSNet/outputs/cvpmvsnet_results/buf/'
# RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/Evaluation/results/CVP-MVSNet/this/'
# METHOD='cvpmvsnet'
# OTHERMSG=''

SET=[1]


mkdir -p $RESULTPATH

matlab -nodesktop -nosplash -r "cd dtu; plyPath='$PLYPATH'; resultsPath='$RESULTPATH'; method_string='$METHOD'; other_msg='$OTHERMSG'; set='$SET'; BaseEvalMain_web"

matlab -nodesktop -nosplash -r "cd dtu; resultsPath='$RESULTPATH'; method_string='$METHOD'; other_msg='$OTHERMSG'; set='$SET'; ComputeStat_web"