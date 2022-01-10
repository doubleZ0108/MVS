import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .mynet import *
import pdb

class FeatureNet(nn.Module):
    """
    in: [3, H, W]
    out: [32, H/4, W/4] (32, 160, 128)
    """
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32
                                                    # in (3, H, W)
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)      # (8, H, W)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)      # (8, H, W)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)     # (16, H/2, W/2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        # 模拟降维中丢掉的信息是回不来的
        # self.conv3_ = ConvBnReLU(16, 16, 5, 2, 2)   # (16, H/4, W/4)
        # self.conv4_ = nn.Sequential(                # (16, H/2, W/2)
        #     nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True))

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)    # (32, H/4, W/4)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)

        # self.conv5_ = ConvBnReLU(16, 32, 3, 1, 1)   # (32, H/2, W/2)
        # self.conv6_ = ConvBnReLU(32, 32, 5, 2, 2)   # (32, H/4, W/4)

        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        # x = self.conv4_(self.conv3_(self.conv2(x)))
        # x = self.feature(self.conv6_(self.conv5_(x)))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x    # (32, 160, 128)


class CostRegNet(nn.Module):
    """
    in: [B, C, D, H/4, W/4] (B, 32, 192, H/4, W/4)
    out: [B, 1, D, H/4, W/4] (B, 1, 192, 160, 128)
    """
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)            # [B, 8, D, H/4, W/4]

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)  # [B, 16, D/2, H/8, W/8]
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2) # [B, 32, D/4, H/16, W/16]
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2) # [B, 64, D/8, H/32, W/32]
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    # @mark 一上来在特征维度降的太多了，而且整体太深了，深度和图像尺寸都被压的太多了，再反卷积回来很多东西都变味了
    def forward(self, x):                       # (B, 32, 192, 160, 128)
        conv0 = self.conv0(x)                   # (B, 8, 192, 160, 128)
        conv2 = self.conv2(self.conv1(conv0))   # (B, 16, 96, 80, 64)
        conv4 = self.conv4(self.conv3(conv2))   # (B, 32, 48, 40, 32)
        x = self.conv6(self.conv5(conv4))       # (B, 64, 24, 20, 16)
        x = conv4 + self.conv7(x)               # (B, 32, 48, 40, 32)
        x = conv2 + self.conv9(x)               # (B, 16, 96, 80, 64)
        x = conv0 + self.conv11(x)              # (B, 8, 192, 160, 128)
        x = self.prob(x)                        # (B, 1, 192, 160, 128)
        return x


class RefineNet(nn.Module):
    """
    in: [B, 4, H/4, W/4] 4是因为img有三通道，depth有1通道
    out: [B, 1, H/4, W/4] (B, 1, 160, 128)
    """
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        # concat = F.cat((img, depth_init), dim=1)
        concat = torch.cat((F.interpolate(img, size=[128, 160]), depth_init.unsqueeze(1)), 1)

        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))   # 原图和深度图拼起来通过网络得到剩余图
        depth_refined = depth_init + depth_residual
        return depth_refined

"""
关于维度:
  B: batch size 在研究数据维度时可以直接将这维去掉
  C: 图像特征维度 最开始是3-channel，后来通过特征提取网络变成32维
  Ndepth: 深度假设维度，这里是192个不同的深度假设
  H: 图像高度，原始是640，经过特征提取网络下采样了四倍，变成160
  W: 图像宽度，同上，512 -> 128
"""
class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

        # self.attnet3d_channel = AttNet3d_channel()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)            # 将[B, 3, 3, H, W]三个三通道图像Tensor -> ([B,3,H,W], [B,3,H,W],[B,3,H,W])三个数据的元组
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]       # 每一个特征[B, 32, H/4, W/4]
            
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)     # [B, 32, 192, H/4, W/4]
        volume_sum = ref_volume             # 将ref的32维特征ref投影过来的高维特征累积构成原始的cost volume
        volume_sq_sum = ref_volume ** 2     # 为了方便后续方差计算
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)     # @mark feature volume [B, C, Ndepth, H, W] 当前HW像素坐标下某个C维的特征，在深度假设下的cost
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume

        # aggregate multiple feature volumes by variance
        # @mark 公式(2) cost volume
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))  # @mark [B, 32, 192, H/4, W/4] 公式(2) 方差简化计算方法  \sum Vi^2 / N - (Vi_bar)^2

        # @doubleZ channel attention
        # cost_att3d_c = self.attnet3d_channel(volume_variance)
        # cost_att3d_c = F.softmax(cost_att3d_c, dim=0)
        # volume_variance = cost_att3d_c * volume_variance
        # if not self.training:
        #     del cost_att3d_c

        # step 3. cost volume regularization
        # @mark 这个cost网络本身是不改变维度的 只是去除噪声更加抽象，真正把32拍成1的是最后一个prob层(soft argmin)，最终的物理含义是 某一个pixel的某一个深度假设位置的概率值
        cost_reg = self.cost_regularization(volume_variance)        # [B, 1, 192, H/4, W/4]
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)              # [B, 192, H/4, W/4] squeeze删除所有维度为1的维度
        prob_volume = F.softmax(cost_reg, dim=1)    # [B, 192, H/4, W/4] 将深度维度的信息压缩为0～1之间的分布，得到概率体probability volume
        depth = depth_regression(prob_volume, depth_values=depth_values)    # 深度图也下采样了四倍 [B, H/4, W/4] 加权平均选择最优的深度 在深度维度做了加权

        with torch.no_grad():
            # photometric confidence @Q 这个不是特别懂，但这几步不影响深度图
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)     # [B, 192, H/4, W/4] 选取最优点周围的四个点平均概率体 最终的尺寸跟概率体是一样的 只不过没处是平均了周围的信息
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()      # [B, H/4, W/4] 这次回归时的深度值是从零开始的整数，最终得到的是index
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)     # [B, H/4, W/4] 从192个深度假设层中获取index对应位置的数据 跟深度图的尺寸是一样的 每点保存的是置信度

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(imgs[0], depth)
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}

# @mark 公式(4)
def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5       # 选择有深度值的gt部分计算loss(mask是二值图像)
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)     # loss = |est - gt|