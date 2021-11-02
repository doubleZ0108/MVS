import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
        

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """投影变换：将src图像的特征投影到ref图像上，融合深度假设范围，得到warped volume

    Args:
        src_fea (src图像的特征)): [B, C, H, W]      此时的C已经是32维了
        src_proj (src图像的投影矩阵): [B, 4, 4]
        ref_proj (参考图像的投影矩阵)): [B, 4, 4]
        depth_values (深度假设范围数组): [B, Ndepth]

    Returns:
        [B, C, Ndepth, H, W] 最终得到的可以理解为src特征图按照不同的深度间隔投影到ref后构建的warped volume
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():       # 阻止梯度计算，降低计算量，保护数据
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))      # src * ref.T
        rot = proj[:, :3, :3]  # [B,3,3] 取左上角三行三列得到旋转变换
        trans = proj[:, :3, 3:4]  # [B,3,1] 取最后一列的上面三行得到平移变换

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])   # 按照ref图像维度构建一张空的平面，之后要做的是根据投影矩阵把src中的像素映射到这张平面上
        y, x = y.contiguous(), x.contiguous()                   # 保证开辟的新空间是连续的(数组存储顺序与按行展开的顺序一致，transpose等操作是跟原tensor共享内存的)
        y, x = y.view(height * width), x.view(height * width)   # 将维度变换为图像样子
        xyz = torch.stack((x, y, torch.ones_like(x)))           # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)       # [B, 3, H*W] unsqueeze先将维度变为[1, 3, H*W], repeat是为了将batch的维度引入进来
        rot_xyz = torch.matmul(rot, xyz)                        # [B, 3, H*W] 先将空白空间乘以旋转矩阵
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) \
                                                    * depth_values.view(batch, 1, num_depth, 1)  # [B, 3, Ndepth, H*W] 再引入Ndepths维度，并将深度假设值填入这个维度
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)                   # [B, 3, Ndepth, H*W] 旋转变换后的矩阵+平移矩阵 -> 投影变换后的空白平面
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]                # [B, 2, Ndepth, H*W] xy分别除以z进行归一
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1         # [B, Ndepth, H*W] x方向按照宽度进行归一
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1        # y方向按照高度进行归一 @Q 这两步不太知道是干什么的？
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)    # [B, Ndepth, H*W, 2] 再把归一化后的x和y拼起来
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), \
                                    mode='bilinear', padding_mode='zeros')      # 按照grid中的映射关系，将src的特征图进行投影变换
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)     # 将上一步编码到height维的深度信息独立出来

    return warped_src_fea



def depth_regression(p, depth_values):
    """深度回归: 根据深度假设和网络的输出计算最终我估计的深度值

    Args:
        p (probability volume): [B, D, H, W]
        depth_values (discrete depth values): [B, D]

    Returns:
        [B, H, W]: 最终估计的深度值
    """
    depth_values = depth_values.view(*depth_values.shape, 1, 1)     # *用来提取数组中的数B和D
    depth = torch.sum(p * depth_values, 1)      # @mark 公式(3) 概率*深度假设 = 期望 -> 估计的最优深度图(是在深度假设维度做的加法，运算后这一维度就没了)
    return depth