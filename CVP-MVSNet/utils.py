# utilities for CVP-MVSNet
# by: Jiayu
# Date: 2019-08-13

# Note: Part of the code is modified from the MVSNet_pytorch project by xy-guo
#       Link: https://github.com/xy-guo/MVSNet_pytorch
#       Thanks the author for such great code!

import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, float):
        return torch.tensor(vars).cuda()
    elif isinstance(vars, np.ndarray):
        return torch.tensor(vars).cuda()
    elif isinstance(vars, int):
        return torch.tensor(vars).cuda()
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))

@make_nograd_func
def Thres_metrics(depth_est, depth_gt, mask, thres):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float()).cpu().numpy()