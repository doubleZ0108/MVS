# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 0018 20:57
# @Author  : Anzhu Yu
# @Site    : 
# @File    : AACVPMVSNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Module import *
import pdb


class ConvBnReLU3D(nn.Module):
    """ConvBnReLU3D
    3D CNN Blocks with batchnorm and activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CostRegNetAACVP(nn.Module):
    def __init__(self, in_channels):
        super(CostRegNetAACVP, self).__init__()
        # 16->in_channels
        self.conv0 = ConvBnReLU3D(in_channels, 16, kernel_size=3, padding=1)
        self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, padding=1)

        self.conv1 = ConvBnReLU3D(16, 32, stride=2, kernel_size=3, padding=1)
        self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, padding=1)
        self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, padding=1)
        self.conv3 = ConvBnReLU3D(32, 64, kernel_size=3, padding=1)
        self.conv4 = ConvBnReLU3D(64, 64, kernel_size=3, padding=1)
        self.conv4a = ConvBnReLU3D(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        self.prob0 = nn.Conv3d(16, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0a(self.conv0(x))
        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4a(self.conv4(self.conv3(conv2)))
        conv5 = conv2 + self.conv5(conv4)
        conv6 = conv0 + self.conv6(conv5)
        prob = self.prob0(conv6).squeeze(1)

        return prob


def sL1_loss(depth_est, depth_gt, mask):
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')


def MSE_loss(depth_est, depth_gt, mask):
    return F.mse_loss(depth_est[mask], depth_gt[mask], size_average=True)


# Here is the network
class AACVPMVSNet(nn.Module):
    def __init__(self, args, group=4, num_heads=1):
        super(AACVPMVSNet, self).__init__()
        self.featurePyramid = FeaturePyramid(num_heads=num_heads)
        self.args = args
        self.Group = group
        self.cost_reg_refine = CostRegNetAACVP(in_channels=self.Group)

    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, depth_min, depth_max):
        # initialization
        depth_est_list = []
        output = {}

        # Step 1: Feature extraction. Self-attention is used here.
        ref_feature_pyramid = self.featurePyramid(ref_img, self.args.nscale)

        src_feature_pyramids = []
        for i in range(self.args.nsrc):
            src_feature_pyramids.append(self.featurePyramid(src_imgs[:, i, :, :, :], self.args.nscale))

        # in. and ex. matrices
        ref_in_multiscales = conditionIntrinsics(ref_in, ref_img.shape,
                                                 [feature.shape for feature in ref_feature_pyramid])
        src_in_multiscales = []
        for i in range(self.args.nsrc):
            src_in_multiscales.append(conditionIntrinsics(src_in[:, i], ref_img.shape,
                                                          [feature.shape for feature in src_feature_pyramids[i]]))
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1, 0, 2, 3, 4)

        # Step 2: estimate the depth map at the coarsest level.
        # nhypothesis = 48 for DTU Dataset as default.
        depth_hypos = calSweepingDepthHypo(ref_in_multiscales[:, -1], src_in_multiscales[:, 0, -1], ref_ex, src_ex,
                                           depth_min, depth_max, nhypothesis_init=48)

        # Step 3: Cost Volume Pyramid calculated here.
        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)

        # @doubleZ TODO
        B, C, H, W = src_feature_pyramids[0][0].shape
        V = self.args.nsrc
        # Kwea3 implementation as reference
        ref_volume = ref_volume.view(B, self.Group, C // self.Group, *ref_volume.shape[-3:])

        volume_sum = 0

        warp_volumes = None
        for src_idx in range(self.args.nsrc):
            # warpped features
            warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:, -1],
                                         src_in_multiscales[:, src_idx, -1, :, :],
                                         ref_ex, src_ex[:, src_idx], depth_hypos)
            ## regular solution
            warped_volume = warped_volume.view(*ref_volume.shape)
            if self.args.mode == "train":
                # (B, Groups, C//Groups, D, h, w)
                volume_sum = volume_sum + warped_volume
            else:
                volume_sum += warped_volume
            del warped_volume

        ## Aggregate multiple feature volumes by Similarity
        ## The parameter V is a little different with that in implementation of Kwea123
        ## V = nsrc here, while V denotes the quantity of all the input images in the implementation of Kwea123.
        cost_volume = (volume_sum * ref_volume).mean(2).div_(V)
        # Step 4: Estimate the Prob.
        cost_reg = self.cost_reg_refine(cost_volume).squeeze(1)

        # Release the GPU burden.
        if self.args.mode == "test":
            del volume_sum
            del ref_volume
            del warp_volumes
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_hypos)
        depth_est_list.append(depth)

        # Step 5: Estimate the residual at each level.
        for level in range(self.args.nscale - 2, -1, -1):
            # Upsample
            depth_up = nn.functional.interpolate(depth[None, :], size=None, scale_factor=2, mode='bicubic',
                                                 align_corners=None)
            depth_up = depth_up.squeeze(0)
            depth_hypos = calDepthHypo(self.args, depth_up, ref_in_multiscales[:, level, :, :],
                                       src_in_multiscales[:, :, level, :, :], ref_ex, src_ex, depth_min, depth_max,
                                       level)
            # @doubleZ TODO 
            cost_volume = proj_cost_AACVP(Group=self.Group, settings=self.args, ref_feature=ref_feature_pyramid[level],
                                          src_feature=src_feature_pyramids,
                                          level=level, ref_in=ref_in_multiscales[:, level, :, :],
                                          src_in=src_in_multiscales[:, :, level, :, :], ref_ex=ref_ex,
                                          src_ex=src_ex[:, :], depth_hypos=depth_hypos)
            cost_reg2 = self.cost_reg_refine(cost_volume).squeeze(1)
            if self.args.mode == "test":
                del cost_volume

            prob_volume = F.softmax(cost_reg2, dim=1)
            if self.args.mode == "test":
                del cost_reg2
            # Depth regression
            depth = depth_regression_refine(prob_volume, depth_hypos)
            depth_est_list.append(depth)

        # Step 6: Get the final result.
        with torch.no_grad():
            num_depth = prob_volume.shape[1]
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                                  dtype=torch.float)).long()
            prob_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        if self.args.mode == "test":
            del prob_volume
            del depth
        ## For T&T and BlendedMVS dataset, the masks are fused with each level at given conf. to avoid noise pixels.
        ## This part is not implemented here.
        ## Return
        depth_est_list.reverse()  # Reverse the list so that depth_est_list[0] is the largest scale.
        output["depth_est_list"] = depth_est_list
        output["prob_confidence"] = prob_confidence

        return output
