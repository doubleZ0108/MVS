PLYPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/MVSNet/outputs/mvsnet_results'
RESULTPATH='/media/public/yan1/doublez/realdoubleZ/Developer/MVS/Evaluation/this/'
METHOD='scan'
OTHERMSG=''


mkdir -p $RESULTPATH

matlab -nodesktop -nosplash -r "cd dtu; plyPath='$PLYPATH'; resultsPath='$RESULTPATH'; method_string='$METHOD'; other_msg='$OTHERMSG'; BaseEvalMain_web"

matlab -nodesktop -nosplash -r "cd dtu; resultsPath='$RESULTPATH'; method_string='$METHOD'; other_msg='$OTHERMSG'; ComputeStat_web"