# MVSNet
# PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/MVSNet/outputs/mvsnet_results/'
# RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/Evaluation/results/MVSNet/firstfulleval/'
# METHOD='mvsnet'
# OTHERMSG='l3'

# CVP-MVSNet
PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/CVP-MVSNet/outputs/cvpmvsnet_results/'
RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/Evaluation/results/CVP-MVSNet/firstfulleval/'
METHOD='cvpmvsnet'
OTHERMSG=''

SET=[1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]


mkdir -p $RESULTPATH

matlab -nodesktop -nosplash -r "cd dtu; plyPath='$PLYPATH'; resultsPath='$RESULTPATH'; method_string='$METHOD'; other_msg='$OTHERMSG'; set='$SET'; BaseEvalMain_web"

matlab -nodesktop -nosplash -r "cd dtu; resultsPath='$RESULTPATH'; method_string='$METHOD'; set='$SET'; ComputeStat_web"