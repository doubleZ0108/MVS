# optimizer
from torch.optim import SGD, Adam
from .optimizers import *
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler
# visualization
from .visualization import *

def get_optimizer(hparams, model):
    eps = 1e-7 if hparams.use_amp else 1e-8
    if hparams.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=hparams.lr, 
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=hparams.lr, eps=eps, 
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=hparams.lr, eps=eps, 
                          weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = Ranger(model.parameters(), lr=hparams.lr, eps=eps, 
                          weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-7 if hparams.use_amp else 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        for k, v in checkpoint['state_dict'].items():
            if not k.startswith('model.'):
                continue
            k = k[6:] # remove 'model.'
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    else: # if it only has model weights
        for k, v in checkpoint.items():
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)