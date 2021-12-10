from models.modules import *
import torch
import torch.nn as nn

class AttNet3d(nn.Module):
    def __init__(self):
        super(AttNet3d, self).__init__()
        self.conv0 = ConvBnReLU3D(16, 16, 3, 1, 1)
        self.conv1 = ConvBnReLU3D(16, 16, 3, 1, 1)
        self.conv2 = ConvBnReLU3D(16, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        att = self.conv2(x)

        return att

class AttNet2d(nn.Module):
    def __init__(self, num_depth):
        super(AttNet2d, self).__init__()
        self.conv0 = ConvBnReLU(num_depth, num_depth, 1, 1, 0)
        self.conv1 = ConvBnReLU(num_depth, num_depth, 1, 1, 0)
        self.conv2 = ConvBnReLU(num_depth, num_depth, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        att = self.conv2(x)

        return att