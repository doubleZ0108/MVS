import os
import shutil


base = '../CVP-MVSNet/outputs/fusibile_fused/'
model_name = 'final3d_model.ply'
target = '../CVP-MVSNet/outputs/cvpmvsnet_results/'
method='cvpmvsnet'
info = 'buf'

if not os.path.exists(target + info):
    os.makedirs(target + info)

for dir in os.listdir(base):
    scan = os.listdir(os.path.join(base, dir))
    index = dir[4:]
    model_dir = [item for item in scan if item.startswith("consistencyCheck")][0]

    old = os.path.join(base, dir, model_dir, model_name)
    fresh = os.path.join(target, info, method) + index.zfile(3) + ".ply"
    print(old)
    print(fresh)
    shutil.copyfile(old, fresh)