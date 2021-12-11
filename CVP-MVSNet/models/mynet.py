from models.modules import *
import torch
import torch.nn as nn

class AttNet3d_channel(nn.Module):
    def __init__(self):
        super(AttNet3d_channel, self).__init__()
        self.conv0 = ConvBnReLU3D(16, 16, 3, 1, 1)
        self.conv1 = ConvBnReLU3D(16, 16, 3, 1, 1)
        self.conv2 = ConvBnReLU3D(16, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        att = self.conv2(x)

        return att

class AttNet3d_depth(nn.Module):
    def __init__(self):
        super(AttNet3d_depth, self).__init__()
        self.conv0 = ConvBnReLU3D(48, 48, 3, 1, 1)
        self.conv1 = ConvBnReLU3D(48, 48, 3, 1, 1)
        self.conv2 = ConvBnReLU3D(48, 1, 3, 1, 1)

        self.conva = ConvBnReLU3D(8, 8, 3, 1, 1)
        self.convb = ConvBnReLU3D(8, 8, 3, 1, 1)
        self.convc = ConvBnReLU3D(8, 1, 3, 1, 1)

    def forward(self, x, num_depth):
        if num_depth == 48:
            x = self.conv0(x)
            x = self.conv1(x)
            att = self.conv2(x)
        elif num_depth == 8:
            x = self.conva(x)
            x = self.convb(x)
            att = self.convc(x)

        return att

class AttNet2d(nn.Module):
    def __init__(self):
        super(AttNet2d, self).__init__()
        self.conv0 = ConvBnReLU(48, 48, 1, 1, 0)
        self.conv1 = ConvBnReLU(48, 48, 1, 1, 0)
        self.conv2 = ConvBnReLU(48, 48, 1, 1, 0)

        self.conva = ConvBnReLU(8, 8, 1, 1, 0)
        self.convb = ConvBnReLU(8, 8, 1, 1, 0)
        self.convc = ConvBnReLU(8, 8, 1, 1, 0)

    def forward(self, x, num_depth):
        if num_depth == 48:
            x = self.conv0(x)
            x = self.conv1(x)
            att = self.conv2(x)
        elif num_depth == 8:
            x = self.conva(x)
            x = self.convb(x)
            att = self.convc(x)

        return att