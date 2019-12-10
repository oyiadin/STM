# coding=utf-8
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CSTM', 'CMM', 'STM']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

def conv1d(in_planes, out_planes):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, padding=1,
                     bias=False)


class _Base(nn.Module):
    def __init__(self, in_channel: int, mid_channel: Optional[int]=None,
                 stride: int=1, time_length: int=8):
        super(_Base, self).__init__()

        out_channel = in_channel

        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.stride = stride
        self.time_length = time_length


class CSTM(_Base):
    def __init__(self, *args, **kwargs):
        super(CSTM, self).__init__(*args, **kwargs)

        self.mid_channel = self.in_channel

        self.conv1d = conv1d(self.in_channel, self.mid_channel)
        quarter = int(0.25 * self.in_channel)
        conv1d_weight = np.zeros_like(self.conv1d.weight.data)
        conv1d_weight[:, 0*quarter:1*quarter, :] = [0.1, 0, 0]
        conv1d_weight[:, 1*quarter:3*quarter, :] = [0, 0.1, 0]
        conv1d_weight[:, 3*quarter:4*quarter, :] = [0, 0, 0.1]
        del self.conv1d.weight
        self.conv1d.weight = nn.Parameter(torch.from_numpy(conv1d_weight))
        self.conv2d = conv3x3(self.mid_channel, self.out_channel,
                              stride=self.stride)

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4, "input should be in (N*T, C, H, W)"
        N, C, H, W = x.size()
        T = self.time_length
        N = int(N / T)

        x = x.reshape((N, T, C, H, W))
        x = x.transpose(1, 4).transpose(2, 3).transpose(1, 2)
        # (N*T, C, H, W) -> (N, T, C, H, W) -> (N, H, W, C, T)
        x = x.reshape((N * H * W, C, T))

        x = self.conv1d(x)

        x = x.reshape((N, H, W, C, T))
        x = x.transpose(1, 2).transpose(2, 3).transpose(1, 4)
        x = x.reshape((N*T, C, H, W))

        x = self.conv2d(x)

        return x


class CMM(_Base):
    def __init__(self, *args, **kwargs):
        super(CMM, self).__init__(*args, **kwargs)

        self.mid_channel = int(self.in_channel / 16)

        self.conv1 = conv1x1(self.in_channel, self.mid_channel)
        self.conv2 = conv3x3(self.mid_channel, self.mid_channel)
        self.bn2 = nn.BatchNorm2d(self.mid_channel)
        if self.stride > 1:
            self.conv3 = conv3x3(self.mid_channel, self.mid_channel,
                                 stride=self.stride)
            self.bn3 = nn.BatchNorm2d(self.mid_channel)
        self.conv4 = conv1x1(self.mid_channel, self.out_channel)

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4, "input should be in (N*T, C, H, W)"
        N, C, H, W = x.size()
        T = self.time_length
        N = int(N / T)
        C = C // 16

        x = self.conv1(x)

        x1 = x[:N*(T-1), :, :, :]
        x2 = x[N:, :, :, :]

        x = self.conv2(x2) - x1
        x = self.bn2(x)

        if self.stride > 1:
            x = self.conv3(x)
            x = self.bn3(x)
            H //= 2
            W //= 2

        x = self.conv4(x)
        C = self.out_channel
        x = x.reshape((N, T-1, C, H, W))
        x = torch.cat((x, torch.zeros((N, 1, C, H, W)).cuda()), dim=1)
        x = x.reshape((N * T, C, H, W))

        return x


class STM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 time_length=None):
        super(STM, self).__init__()
        assert time_length is not None, "time_length is required"

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.cstm = CSTM(planes, stride=stride, time_length=time_length)
        self.cmm = CMM(planes, stride=stride, time_length=time_length)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.time_length = time_length
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4, "input should be in (N*T, C, H, W)"
        identity = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        cstm = self.cstm(x)
        cmm = self.cmm(x)
        x = cstm + cmm
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + identity
        x = self.relu(x)

        return x
