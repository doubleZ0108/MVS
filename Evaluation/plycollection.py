import os
import shutil

def collect_CVPMVSNet():
    thisname = 'att3_full_22'
    base = os.path.join('outputs/CVP-MVSNet/', thisname, 'fusibile_fused')
    model_name = 'final3d_model.ply'
    target = os.path.join('outputs/CVP-MVSNet/', thisname, 'cvpmvsnet_results')
    method='cvpmvsnet'
    info = ''

    if not os.path.exists(target + info):
        os.makedirs(target + info)

    for dir in os.listdir(base):
        scan = os.listdir(os.path.join(base, dir))
        index = dir[4:]
        model_dir = [item for item in scan if item.startswith("consistencyCheck")][0]

        old = os.path.join(base, dir, model_dir, model_name)
        fresh = os.path.join(target, info, method) + index.zfill(3) + ".ply"
        # shutil.copyfile(old, fresh)
        shutil.move(old, fresh)

    print("ply moving done!")

if __name__ == '__main__':
    collect_CVPMVSNet()