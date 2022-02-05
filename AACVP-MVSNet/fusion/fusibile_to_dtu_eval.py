# Convert output of fusibile to DTU evaluation format.
# Author: Jiayu Yang
# Modified by B. Liu

import os
from os import listdir

fusibile_out_folder="../outputs/fusibile_fused/"
dtu_eval_folder="../outputs/"

if not os.path.isdir(dtu_eval_folder):
    os.mkdir(dtu_eval_folder)

# Read test list
testlist = "./scan_list_test.txt"
with open(testlist) as f:
    scans = f.readlines()
    scans = [line.rstrip() for line in scans]

for scan in scans:
    # Move ply to dtu eval folder and rename
    scan_folder = os.path.join(fusibile_out_folder,scan)
    consis_folders = [f for f in listdir(scan_folder) if f.startswith('consistencyCheck-')]
    consis_folders.sort()
    consis_folder = consis_folders[-1]
    source_ply = os.path.join(fusibile_out_folder,scan,consis_folder,'final3d_model.ply')
    scan_idx = int(scan[4:])
    target_ply = os.path.join(dtu_eval_folder,'AACVPMVSnet{:03d}_l3.ply'.format(scan_idx))

    cmd = 'cp '+source_ply+' '+target_ply

    print(cmd)
    os.system(cmd)
