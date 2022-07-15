from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *


class DBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=1):
        super(DBL, self).__init__()
        if padding:
            padding = (kernel - 1) // 2
        else:
            padding = 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.dbl1 = DBL(in_channels, in_channels // 2, 1, 1)
        self.dbl2 = DBL(in_channels // 2, out_channels, 3, 1)

    def forward(self, x):
        out = x
        out = self.dbl1(out)
        out = self.dbl2(out)
        out = out + x
        return out


class ResX(nn.Module):
    def __init__(self, num_resunit, in_channels):
        super(ResX, self).__init__()
        self.dbl = DBL(in_channels, in_channels * 2, 3, 2)
        self.resunit = ResUnit(in_channels * 2, in_channels * 2)
        self.num_resunit = num_resunit

    def forward(self, x):
        x = self.dbl(x)
        for i in range(self.num_resunit):
            x = self.resunit(x)
        return x


class Darknet(nn.Module):
    def __init__(self,
                 nums_resunits=[1, 2, 8, 8, 4],
                 nums_channels=[32, 64, 128, 256, 512]):
        super(Darknet, self).__init__()
        self.first_dbl = DBL(3, nums_channels[0], 3, 1)
        self.resx = nn.ModuleList()
        for idx, num in enumerate(nums_resunits):
            self.resx.append(ResX(num, in_channels=nums_channels[idx]))

    def forward(self, x):
        x = self.first_dbl(x)
        output = []
        for i in range(len(self.resx)):
            x = self.resx[i](x)
            if i >= 2:
                output.append(x)
        return output


class MultiDBL(nn.Module):
    def __init__(self, in_channels, hiden_channels):
        super(MultiDBL, self).__init__()
        self.dbl1 = DBL(in_channels, hiden_channels, 1, 1)
        self.dbl2 = DBL(hiden_channels, hiden_channels * 2, 3, 1)
        self.dbl3 = DBL(hiden_channels * 2, hiden_channels, 1, 1)
        self.dbl4 = DBL(hiden_channels, hiden_channels * 2, 3, 1)
        self.dbl5 = DBL(hiden_channels * 2, hiden_channels, 1, 1)

    def forward(self, x):
        x = self.dbl1(x)
        x = self.dbl2(x)
        x = self.dbl3(x)
        x = self.dbl4(x)
        x = self.dbl5(x)
        return x


class FPN(nn.Module):
    def __init__(self, num_classes, in_channels=1024):
        super(FPN, self).__init__()
        self.multi_dbl1 = MultiDBL(1024, 512)
        self.multi_dbl2 = MultiDBL(768, 256)  # 512+256=768
        self.multi_dbl3 = MultiDBL(384, 128)  # 256+128=384

        self.dbl1 = nn.Sequential(DBL(in_channels // 2, in_channels, 3, 1),
                                  nn.Conv2d(in_channels, (num_classes + 5) * 3, 1, 1))
        self.dbl2 = nn.Sequential(DBL(in_channels // 4, in_channels // 2, 3, 1),
                                  nn.Conv2d(in_channels // 2, (num_classes + 5) * 3, 1, 1))
        self.dbl3 = nn.Sequential(DBL(in_channels // 8, in_channels // 4, 3, 1),
                                  nn.Conv2d(in_channels // 4, (num_classes + 5) * 3, 1, 1))

        self.upsample_block1 = nn.Sequential(
            DBL(in_channels // 2, in_channels // 4, 1, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.upsample_block2 = nn.Sequential(
            DBL(in_channels // 4, in_channels // 8, 1, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

    def forward(self, xin):
        outputs = []
        output1_temp = self.multi_dbl1(xin[2])
        # output1 = self.conv(self.dbl1(output1_temp))
        outputs.append(self.dbl1(output1_temp))

        output2 = self.upsample_block1(output1_temp)
        output2 = torch.cat([output2, xin[1]], 1)
        output2_temp = self.multi_dbl2(output2)
        # output2 = self.conv(self.dbl(output2_temp))
        outputs.append(self.dbl2(output2_temp))

        output3 = self.upsample_block2(output2_temp)
        output3 = torch.cat([output3, xin[0]], 1)
        output3 = self.multi_dbl3(output3)
        # output3 = self.conv(self.dbl(output3))
        outputs.append(self.dbl3(output3))

        return outputs


def create_model(num_classes):
    return nn.Sequential(Darknet([1, 2, 8, 8, 4], [32, 64, 128, 256, 512]),
                         FPN(num_classes))
