#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 0016 11:52
# @Author  : Anzhu Yu
# @Site    : 
# @File    : train_AACVPMVSNet.py
# @Software: PyCharm

# some packages used in this project
from genericpath import exists
from argsParser import getArgsParser, checkArgs
import os
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from datasets import dtu_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from models import *
from utils import *
import gc
import sys
import datetime
import torch.utils
import torch.utils.checkpoint

# CUDA_LAUNCH_BLOCKING=1


parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "train", 'HERE IS THE TRAINING MODE！'
checkArgs(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
log_path = args.loggingdir + args.info.replace(" ", "_") + "/"
if not os.path.isdir(args.loggingdir):
    os.mkdir(args.loggingdir)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + curTime + '.log'
logfile = log_name
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fileHandler = logging.FileHandler(logfile, mode='a')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.info("Logger initialized.")
logger.info("Writing logs to file:" + logfile)

settings_str = "All settings:\n"
line_width = 20
for k, v in vars(args).items():
    settings_str += '{0}: {1}\n'.format(k, v)
logger.info(settings_str)

# Read the Data,
train_dataset = dtu_loader.MVSDataset(args, logger)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)

# Build the model
model = AACVPMVSNet(args, group=args.groups, num_heads=args.num_heads)

# Use the cuda_ids to determine the GPUs used for experiments
device_id_list = [int(idd) for idd in args.cuda_ids.split(',')]

if len(device_id_list) == 1 and (device_id_list[0] != 666):
    print("Now multi-GPUs mode activated!")
    device_ids = [int(args.cuda_ids)]
elif (device_id_list[0] == 666):
    model = model.cpu()
else:
    device_ids = device_id_list
del device_id_list

# GPUs
# @doubleZ TODO 
if args.mode == "train" and torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
if torch.cuda.is_available():
    model.cuda()
model.train()


if args.loss_function == "sl1":
    logger.info("Using smoothed L1 loss")
    model_loss = sL1_loss
else:  # MSE
    logger.info("Using MSE loss")
    model_loss = MSE_loss
logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
# model_loss = mvsnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# Start at a given checkpoint
sw_path = args.logckptdir + args.info + "/"
os.makedirs(sw_path, exist_ok=True)
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    logger.info("Resuming or testing...")
    saved_models = [fn for fn in os.listdir(sw_path) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # use the latest checkpoint file
    loadckpt = os.path.join(sw_path, saved_models[-1])
    logger.info("Resuming " + loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# Training at each epoch
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    # epoch stat
    last_loss = None
    this_loss = None
    for epoch_idx in range(start_epoch, args.epochs):
        logger.info('Epoch {}:'.format(epoch_idx))
        global_step = len(train_loader) * epoch_idx

        if last_loss is None:
            last_loss = 999999
        else:
            last_loss = this_loss
        this_loss = []

        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss = train_sample(sample, detailed_summary=do_summary)
            this_loss.append(loss)

            logger.info(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(train_loader), loss,
                                                                                     time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logckptdir + args.info.replace(" ", "_"), epoch_idx))
            logger.info("model_{:0>6}.ckpt saved".format(epoch_idx))
        this_loss = np.mean(this_loss)
        logger.info("Epoch loss: {:.5f} --> {:.5f}".format(last_loss, this_loss))

        lr_scheduler.step()


# Training for each batch
def train_sample(sample, detailed_summary=False):
    """
    :param sample: each batch
    :param detailed_summary: whether the detailed logs are needed.
    :return: the loss
    """
    # model.train() is not needed here, however it is often used to state this script is not for evaluation.
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    ref_depths = sample_cuda["ref_depths"]

    # forward
    outputs = model(sample_cuda["ref_img"].float(), sample_cuda["src_imgs"].float(), sample_cuda["ref_intrinsics"],
                    sample_cuda["src_intrinsics"], sample_cuda["ref_extrinsics"], sample_cuda["src_extrinsics"],
                    sample_cuda["depth_min"], sample_cuda["depth_max"])

    depth_est_list = outputs["depth_est_list"]

    dHeight = ref_depths.shape[2]
    dWidth = ref_depths.shape[3]
    loss = []
    for i in range(0, args.nscale):
        # generate the masks.
        depth_gt = ref_depths[:, i, :int(dHeight / 2 ** i), :int(dWidth / 2 ** i)]
        mask = depth_gt > 425
        loss.append(model_loss(depth_est_list[i], depth_gt.float(), mask))

    loss = sum(loss)

    loss.backward()

    optimizer.step()

    return loss.data.cpu().item()


# main function, the start of this program
if __name__ == '__main__':
    if args.mode == "train":
        train()
