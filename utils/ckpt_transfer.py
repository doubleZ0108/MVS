'''
script for solving the following issue:
"xxx.ckpt is a zip archive(did you mean to use torch.jit.load()?)"
'''

import os
import torch    # torch >= 1.6

basedir = './checkpoints/baseline/train_dtu_128/'
targetdir = './checkpoints/save/train_dtu'

for file in os.listdir(basedir):
    thisckpt = os.path.join(basedir, file)
    state_dict = torch.load(thisckpt)
    torch.save(state_dict, os.path.join(targetdir, file), _use_new_zipfile_serialization=False)

# num = 39
# thisckpt = "./checkpoints/baseline/train_dtu_128/model_" + str(num).zfill(6) + ".ckpt"
# state_dict = torch.load(thisckpt)
# torch.save(state_dict, "./checkpoints/save/train_dtu/model_" + str(num).zfill(6) + ".ckpt", _use_new_zipfile_serialization=False)
