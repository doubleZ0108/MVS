from .module import *
import torch.nn as nn

class AttNet3d_channel(nn.Module):
    def __init__(self):
        super(AttNet3d_channel, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 32, 3, 1, 1)
        self.conv1 = ConvBnReLU3D(32, 32, 3, 1, 1)
        self.conv2 = ConvBnReLU3D(32, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        att = self.conv2(x)

        return att