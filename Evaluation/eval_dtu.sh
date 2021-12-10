# MVSNet
# THISNAME='outputs_baseline_full_22'
# PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/MVSNet/'$THISNAME'/mvsnet_results/'
# RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/results/MVSNet/'$THISNAME'/'
# METHOD='mvsnet'
# OTHERMSG='l3'

# CVP-MVSNet
THISNAME='baseline_full_22'
PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/outputs/CVP-MVSNet/'$THISNAME'/cvpmvsnet_results/'
RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/results/CVP-MVSNet/'$THISNAME'/'
LOGPATH='/media/public/yan1/doublez/realdoubleZ/Developer/Evaluation/results/CVP-MVSNet/'$THISNAME'/'$THISNAME'.log'

METHOD='cvpmvsnet'
OTHERMSG=''

SET=[1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
# SET=[1]

mkdir -p $RESULTPATH

matlab -nodesktop -nosplash -r "cd dtu; plyPath='$PLYPATH'; resultsPath='$RESULTPATH'; method_string='$METHOD'; other_msg='$OTHERMSG'; set='$SET'; BaseEvalMain_web" > $LOGPATH

matlab -nodesktop -nosplash -r "cd dtu; resultsPath='$RESULTPATH'; method_string='$METHOD'; set='$SET'; ComputeStat_web" >> $LOGPATH