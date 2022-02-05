# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 0018 20:57
# @Author  : Anzhu Yu
# @Site    : 
# @File    : module.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def homo_warping(src_feature, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    # Apply homography warpping on one src feature map from src to ref view.

    batch, channels = src_feature.shape[0], src_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = src_feature.shape[2], src_feature.shape[3]
    with torch.no_grad():
        src_proj = torch.matmul(src_in, src_ex[:, 0:3, :])
        ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
        last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(len(src_in), 1, 1).cuda()
        src_proj = torch.cat((src_proj, last), 1)
        ref_proj = torch.cat((ref_proj, last), 1)
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,
                                                                                           1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea


def depth_regression(p, depth_values):
    """
    :param p: probability volume [B, D, H, W]
    :param depth_values: discrete depth values [B, D]
    :return: depth
    """
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

# Self-attention layer
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # make sure that out_channels = 0 (mod groups)
        assert self.out_channels % self.groups == 0, "ERROR INPUT,CHECK AGAIN!"
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # Learned transformation.
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # Activation here. The same with all the other conv layers.
        return nn.LeakyReLU(0.1)(out)

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


## General convolution
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


# Feature Extraction
class FeaturePyramid(nn.Module):
    def __init__(self, num_heads=1):
        super(FeaturePyramid, self).__init__()
        self.conv0aa = conv(3,  64, kernel_size=3, stride=1)
        self.conv0ba = conv(64, 64, kernel_size=3, stride=1)
        self.conv0bb = conv(64, 64, kernel_size=3, stride=1)
        self.conv0bc = conv(64, 32, kernel_size=3, stride=1)
        self.conv0bd = conv(32, 32, kernel_size=3, stride=1)
        self.conv0be = conv(32, 32, kernel_size=3, stride=1)
        self.conv0bf = conv(32, 16, kernel_size=3, stride=1)
        self.conv0bg = conv(16, 16, kernel_size=3, stride=1)
        self.conv0bh = AttentionConv(16, 16, kernel_size=3, stride=1, groups=num_heads)

    def forward(self, img, scales=5):
        fp = []
        f = self.conv0aa(img)
        f = self.conv0bh(
            self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
        fp.append(f)
        for scale in range(scales - 1):
            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=None).detach()
            f = self.conv0aa(img)
            f = self.conv0bh(
                self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
            fp.append(f)

        return fp


def conditionIntrinsics(intrinsics, img_shape, fp_shapes):
    # Pre-condition intrinsics according to feature pyramid shape.
    # Calculate downsample ratio for each level of feature pyramid
    down_ratios = []
    for fp_shape in fp_shapes:
        down_ratios.append(img_shape[2] / fp_shape[2])

    # condition intrinsics
    intrinsics_out = []
    for down_ratio in down_ratios:
        intrinsics_tmp = intrinsics.clone()
        # print(down_ratio)
        intrinsics_tmp[:, :2, :] = intrinsics_tmp[:, :2, :] / down_ratio
        intrinsics_out.append(intrinsics_tmp)

    return torch.stack(intrinsics_out).permute(1, 0, 2, 3)  # [B, nScale, 3, 3]


def calInitDepthInterval(ref_in, src_in, ref_ex, src_ex, pixel_interval):
    return 165  # The mean depth interval calculated on 4-1 interval setting...

def calSweepingDepthHypo(ref_in, src_in, ref_ex, src_ex, depth_min, depth_max, nhypothesis_init=48):
    # Batch
    batchSize = ref_in.shape[0]
    depth_range = depth_max[0] - depth_min[0]
    depth_interval_mean = depth_range / (nhypothesis_init - 1)
    # Make sure the number of depth hypothesis has a factor of 2
    assert nhypothesis_init % 2 == 0

    depth_hypos = torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)

    # Assume depth range is consistent in one batch.
    for b in range(1, batchSize):
        depth_range = depth_max[b] - depth_min[b]
        depth_hypos = torch.cat(
            (depth_hypos, torch.range(depth_min[0], depth_max[0], depth_interval_mean).unsqueeze(0)), 0)

    return depth_hypos.cuda()


def calDepthHypo(netArgs, ref_depths, ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics, depth_min,
                 depth_max, level):
    ## Calculate depth hypothesis maps for refine steps

    # These two parameters determining the depth searching range and interval at finer level.
    # For experiments on other datasets, the pixel_interval could be modified accordingly to get better results.
    d = 4
    pixel_interval = 1

    nBatch = ref_depths.shape[0]
    height = ref_depths.shape[1]
    width = ref_depths.shape[2]
    # Hard code the interval for training on DTU with 1 level of refinement.
    # This depth interval is estimated by J.Yang for training boosting.
    # Uncomment this part if other dataset is used.
    if netArgs.mode == "train":

        depth_interval = torch.tensor(
            [6.8085] * nBatch).cuda()
        depth_hypos = ref_depths.unsqueeze(1).repeat(1, d * 2, 1, 1)
        # print(depth_interval[0])
        for depth_level in range(-d, d):
            depth_hypos[:, depth_level + d, :, :] += (depth_level) * depth_interval[0]
        return depth_hypos

    with torch.no_grad():
        ref_depths = ref_depths
        ref_intrinsics = ref_intrinsics.double()
        src_intrinsics = src_intrinsics.squeeze(1).double()
        ref_extrinsics = ref_extrinsics.double()
        src_extrinsics = src_extrinsics.squeeze(1).double()

        interval_maps = []
        depth_hypos = ref_depths.unsqueeze(1).repeat(1, d * 2, 1, 1)
        for batch in range(nBatch):
            xx, yy = torch.meshgrid([torch.arange(0, width).cuda(), torch.arange(0, height).cuda()])

            xxx = xx.reshape([-1]).double()
            yyy = yy.reshape([-1]).double()

            X = torch.stack([xxx, yyy, torch.ones_like(xxx)], dim=0)

            D1 = torch.transpose(ref_depths[batch, :, :], 0, 1).reshape(
                [-1])  # Transpose before reshape to produce identical results to numpy and matlab version.
            D2 = D1 + 1

            X1 = X * D1.double()
            X2 = X * D2.double()
            ray1 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X1)
            ray2 = torch.matmul(torch.inverse(ref_intrinsics[batch]), X2)

            X1 = torch.cat([ray1, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X1 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X1)
            X2 = torch.cat([ray2, torch.ones_like(xxx).unsqueeze(0).double()], dim=0)
            X2 = torch.matmul(torch.inverse(ref_extrinsics[batch]), X2)

            X1 = torch.matmul(src_extrinsics[batch][0], X1)
            X2 = torch.matmul(src_extrinsics[batch][0], X2)

            X1 = X1[:3]
            X1 = torch.matmul(src_intrinsics[batch][0], X1)
            X1_d = X1[2].clone()
            X1 /= X1_d

            X2 = X2[:3]
            X2 = torch.matmul(src_intrinsics[batch][0], X2)
            X2_d = X2[2].clone()
            X2 /= X2_d

            k = (X2[1] - X1[1]) / (X2[0] - X1[0])
            b = X1[1] - k * X1[0]

            theta = torch.atan(k)
            X3 = X1 + torch.stack(
                [torch.cos(theta) * pixel_interval, torch.sin(theta) * pixel_interval, torch.zeros_like(X1[2, :])],
                dim=0)

            A = torch.matmul(ref_intrinsics[batch], ref_extrinsics[batch][:3, :3])
            tmp = torch.matmul(src_intrinsics[batch][0], src_extrinsics[batch][0, :3, :3])
            A = torch.matmul(A, torch.inverse(tmp))

            tmp1 = X1_d * torch.matmul(A, X1)
            tmp2 = torch.matmul(A, X3)

            M1 = torch.cat([X.t().unsqueeze(2), tmp2.t().unsqueeze(2)], axis=2)[:, 1:, :]
            M2 = tmp1.t()[:, 1:]
            ans = torch.matmul(torch.inverse(M1), M2.unsqueeze(2))
            delta_d = ans[:, 0, 0]

            interval_maps = torch.abs(delta_d).mean().repeat(ref_depths.shape[2], ref_depths.shape[1]).t().float()

            for depth_level in range(-d, d):
                depth_hypos[batch, depth_level + d, :, :] += depth_level * interval_maps

        return depth_hypos.float()  # Return the depth hypothesis map from statistical interval setting.



def depth_regression_refine(prob_volume, depth_hypothesis):
    depth = torch.sum(prob_volume * depth_hypothesis, 1)
    return depth


def proj_cost_AACVP(Group, settings, ref_feature, src_feature, level, ref_in, src_in, ref_ex, src_ex, depth_hypos):
    ## Calculate the cost volume for refined depth hypothesis selection
    # AACVP Version.
    batch, channels = ref_feature.shape[0], ref_feature.shape[1]
    num_depth = depth_hypos.shape[1]
    height, width = ref_feature.shape[2], ref_feature.shape[3]
    B, C, H, W = ref_feature.shape
    volume_sum = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    ref_volume = volume_sum
    ref_volume = ref_volume.view(B, Group, C // Group, *ref_volume.shape[-3:])
    volume_sum = 0
    for src in range(settings.nsrc):
        with torch.no_grad():
            src_proj = torch.matmul(src_in[:, src, :, :], src_ex[:, src, 0:3, :])
            ref_proj = torch.matmul(ref_in, ref_ex[:, 0:3, :])
            last = torch.tensor([[[0, 0, 0, 1.0]]]).repeat(len(src_in), 1, 1).cuda()
            src_proj = torch.cat((src_proj, last), 1)
            ref_proj = torch.cat((ref_proj, last), 1)

            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_feature.device),
                                   torch.arange(0, width, dtype=torch.float32, device=ref_feature.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
            rot_xyz = torch.matmul(rot, xyz)

            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_hypos.view(batch, 1, num_depth,
                                                                                               height * width)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
            grid = proj_xy

        warped_src_fea = F.grid_sample(src_feature[src][level], grid.view(batch, num_depth * height, width, 2),
                                       mode='bilinear',
                                       padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
        warped_src_fea = warped_src_fea.to(ref_volume.dtype)

        warped_src_fea = warped_src_fea.view(*ref_volume.shape)
        if settings.mode == 'training':
            volume_sum = volume_sum + warped_src_fea  # (B, Group, C//Group, D, h, w)
        else:
            volume_sum += warped_src_fea
        del warped_src_fea

    volume_variance = (volume_sum * ref_volume).mean(2).div_(settings.nsrc)  # (B, Group, D, h, w)
    del volume_sum, ref_volume

    return volume_variance
