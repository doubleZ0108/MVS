#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 0028 11:52
# @Author  : Anzhu Yu (Modified). The original
# @Site    :
# @File    : train.py
# @Software: PyCharm


import argparse

def getArgsParser():
    parser = argparse.ArgumentParser(description='AACVP-MVSNet. Thanks J.Yang and X.Guo for sharing such good projects as references.')

    # The 'general settings' and 'training settings' are the same with those of  J.Yang for fair comparson,
    # while the 'epoch' parameter is set to 40 instead of 28.
    # General settings
    parser.add_argument('--info', default='None', help='Info about current run')
    parser.add_argument('--mode', default='train', help='train or test ro validation', choices=['train', 'test', 'val'])
    parser.add_argument('--dataset_root', default='./datasets/dataset/dtu-train-128/',help='path to dataset root')
    parser.add_argument('--imgsize', type=int, default=128, choices=[128,1200], help='height of input image')
    parser.add_argument('--nsrc', type=int, default=3, help='number of src views to use')
    parser.add_argument('--nscale', type=int, default=7, help='number of scales to use')

    # Training settings
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrepochs', type=str, default="10,12,14,20:2", help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--loss_function', default='sl1', help='which loss function to use', choices=['sl1','mse'])
    
    # Checkpoint settings
    parser.add_argument('--loadckpt', type=str, default='', help='load a specific checkpoint')
    parser.add_argument('--logckptdir', default='./checkpoints/', help='the directory to save checkpoints/logs')
    parser.add_argument('--loggingdir', default='./logs/', help='the directory to save logging outputs')
    parser.add_argument('--resume', type=int, default=0, help='continue to train the model')
    
    # Evaluation settings 
    parser.add_argument('--outdir', default='./outputs/debug/', help='the directory to save depth outputs')
    parser.add_argument('--eval_visualizeDepth', type=int, default=1)
    parser.add_argument('--eval_prob_filtering', type=int, default=0)
    parser.add_argument('--eval_prob_threshold', type=float, default=0.99)
    parser.add_argument('--eval_shuffle', type=int, default=0)

    # Here is new parameters
    # Since 4 2080ti(s) are used for training in our paper, this parameter should be modified  according to the equipments used.
    parser.add_argument('--cuda_ids', default='0,1,2,3', help="GPUs used in train or test，'0,1' for example."
                                                              "If only cpu is used, use '666' instead")
    parser.add_argument('--groups', type=int, default=4,help='Groups used for GWC.')
    parser.add_argument('--num_heads',type=int, default=1, help='Heads for Self-Attention layer. Single head is set as default.')

    return parser


def checkArgs(args):
    # Check if the settings is valid
    assert args.mode in ["train", "val", "test"]
    if args.resume:
        assert len(args.loadckpt) == 0
    if args.loadckpt:
        assert args.resume is 0