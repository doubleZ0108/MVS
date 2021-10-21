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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out

# non-use
class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """投影变换：将src图像的特征投影到ref图像上，融合深度假设范围，得到warped volume

    Args:
        src_fea (src图像的特征)): [B, C, H, W]
        src_proj (src图像的投影矩阵): [B, 4, 4]
        ref_proj (参考图像的投影矩阵)): [B, 4, 4]
        depth_values (深度假设范围数组): [B, Ndepth]

    Returns:
        [B, C, Ndepth, H, W] 最终得到的可以理解为src特征图按照不同的深度间隔投影到ref后构建的warped特征体
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():       # 阻止梯度计算，降低计算量，保护数据
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))  
        rot = proj[:, :3, :3]  # [B,3,3] 取左上角三行三列得到投影变换
        trans = proj[:, :3, 3:4]  # [B,3,1] 取最后一列的上面三行得到平移变换

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])     # 按照ref图像维度构建一张空的平面，之后要做的是根据投影矩阵把src中的像素映射到ref上
        y, x = y.contiguous(), x.contiguous()                   # 保证开辟的新空间是连续的(数组存储顺序与按行展开的顺序一致，transpose等操作是跟原tensor共享内存的)
        y, x = y.view(height * width), x.view(height * width)   # 将维度变换为图像样子
        xyz = torch.stack((x, y, torch.ones_like(x)))           # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)       # [B, 3, H*W] unsqueeze先将维度变为[1, 3, H*W], repeat是为了将batch的维度引入进来
        rot_xyz = torch.matmul(rot, xyz)                        # [B, 3, H*W] 先将空白空间乘以旋转矩阵
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) \
                                                    * depth_values.view(batch, 1, num_depth, 1)  # [B, 3, Ndepth, H*W] 再引入Ndepths维度，并将深度假设值填入这个维度
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)                   # [B, 3, Ndepth, H*W] 旋转变换后的矩阵+平移矩阵 -> 投影变换后的空白平面
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]                # [B, 2, Ndepth, H*W] xy分别除以z进行归一
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1         # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1        # @Q
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)    # [B, Ndepth, H*W, 2] 再把归一化后的x和y拼起来
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), \
                                    mode='bilinear', padding_mode='zeros')      # 按照grid中的映射关系，将src的特征图进行投影变换
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

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


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
