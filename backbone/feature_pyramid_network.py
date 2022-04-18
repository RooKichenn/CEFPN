from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


def xavier_init(m, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(m.weight, gain=gain)
    else:
        nn.init.xavier_normal_(m.weight, gain=gain)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, bias)


class SCE(nn.Module):
    def __init__(self, in_channels):
        super(SCE, self).__init__()
        # C = 2048
        # ----------------------------------------------------- #
        # 第一个分支  w, h, C --> w, h, C/2 --> SSF --> 2w, 2h, C
        # ----------------------------------------------------- #
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        # ----------------------------------------------------- #
        # 第二个分支  w, h, C --> w/2, h/2, 2C --> SSF --> 2w, 2h, C
        # ----------------------------------------------------- #
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        self.pixel_shuffle_4 = nn.PixelShuffle(upscale_factor=4)

        # ----------------------------------------------------- #
        # 第三个分支  w, h, C --> 1, 1, C --> broadcast
        # ----------------------------------------------------- #
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_3 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        out_size = x.shape[-2:]
        out_size = [x*2 for x in out_size]
        branch1 = self.pixel_shuffle(self.conv3x3(x))
        branch2 = F.interpolate(self.pixel_shuffle_4(self.conv1x1_2(self.maxpool(x))), size=out_size, mode="nearest")
        branch3 = self.conv1x1_3(self.globalpool(x))
        out = (branch1 + branch2 + branch3)
        return out


class CAG(nn.Module):
    def __init__(self, in_channels):
        super(CAG, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channels, in_channels, 1)
        self.fc2 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        fc1 = self.sigmoid(self.fc1(self.avgpool(x)))
        fc2 = self.fc2(self.maxpool(x))
        out = fc1 + fc2
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """
    def __init__(self, in_channels, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.extra_blocks = extra_blocks
        # 亚像素上采样，scale默认是2
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        # ------------------------------- #
        # 定义SCE模块
        # ------------------------------- #
        self.SCE = SCE(in_channels=in_channels)

        # ------------------------------- #
        # 定义CAG模块
        # ------------------------------- #
        self.CAG = CAG(in_channels=in_channels // 8)

        # ------------------------------- #
        # 定义1x1卷积
        # ------------------------------- #

        # 经过SSF后的1x1卷积
        self.SSF_C5 = nn.Conv2d(512, 256, 1)
        self.SSF_C4 = nn.Conv2d(256, 256, 1)

        # ------------------------------- #
        # 定义Ci --> Fi 的1x1卷积
        # ------------------------------- #

        self.conv_1x1_4 = nn.Conv2d(1024, 256, 1)
        self.conv_1x1_3 = nn.Conv2d(512, 256, 1)
        self.conv_1x1_2 = nn.Conv2d(256, 256, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        # ------------------------------- #
        # 获得Ci特征层
        # ------------------------------- #
        C2, C3, C4, C5 = x

        # ------------------------------- #
        # 得到SCE模块的输出
        # ------------------------------- #
        SCE_out = self.SCE(C5)

        # ------------------------------- #
        # 得到Fi特征层
        # ------------------------------- #
        F4 = self.SSF_C5(self.pixel_shuffle(C5)) + self.conv_1x1_4(C4)
        F3 = self.SSF_C4(self.pixel_shuffle(C4)) + self.conv_1x1_3(C3)
        F2 = self.conv_1x1_2(C2)

        # ------------------------------- #
        # 得到Pi特征层
        # ------------------------------- #
        P4 = F4
        P4_upsample = F.interpolate(P4, size=F3.shape[-2:], mode='nearest')
        P3 = F3 + P4_upsample
        P3_upsample = F.interpolate(P3, size=F2.shape[-2:], mode="nearest")
        P2 = F2 + P3_upsample

        # ------------------------------- #
        # 得到特征图I
        # ------------------------------- #
        out_size = P3.shape[-2:]
        SCE_out = F.interpolate(SCE_out, size=out_size, mode="nearest")
        I_P4 = F.interpolate(P4, size=out_size, mode="nearest")
        I_P3 = F.adaptive_max_pool2d(P3, output_size=out_size)
        I_P2 = F.adaptive_max_pool2d(P2, output_size=out_size)

        I = (I_P4 + I_P3 + I_P2 + SCE_out) / 4

        # ------------------------------- #
        # 得到特征图Ri和CA
        # ------------------------------- #
        outs = []
        CA = self.CAG(I)
        R5 = F.adaptive_max_pool2d(I, output_size=C5.shape[-2:])
        R5 = R5 * CA
        residual_R4 = F.adaptive_max_pool2d(I, output_size=C4.shape[-2:])
        R4 = (residual_R4 + F4) * CA
        residual_R3 = F.interpolate(I, size=C3.shape[-2:], mode="nearest")
        R3 = (residual_R3 + F3) * CA
        residual_R2 = F.interpolate(I, size=C2.shape[-2:], mode="nearest")
        R2 = (residual_R2 + F2) * CA
        for i in [R2, R3, R4, R5]:
            outs.append(i)

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            outs, names = self.extra_blocks(outs, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, outs)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names
