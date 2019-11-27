# coding=utf-8
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CSTM', 'CMM', 'STM']


class _Base(nn.Module):
    def __init__(self, in_channel: int, mid_channel: Optional[int]=None,
                 stride: int=1, time_length: int = 3,
                 conv1d_settings: Optional[dict] = None,
                 conv2d_settings: Optional[dict] = None):
        super(_Base, self).__init__()

        out_channel = in_channel

        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.stride = stride
        self.time_length = time_length
        self.conv1d_settings = conv1d_settings
        self.conv2d_settings = conv2d_settings


class CSTM(_Base):
    def __init__(self, *args, **kwargs):
        super(CSTM, self).__init__(*args, **kwargs)

        self.mid_channel = self.in_channel

        self.conv1d = nn.Conv1d(self.in_channel, self.mid_channel,
                                kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(self.mid_channel, self.out_channel,
                                kernel_size=(3, 3),
                                padding=(1, 1),
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

        self.conv1 = nn.Conv2d(self.in_channel, self.mid_channel,
                               kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(self.mid_channel, self.mid_channel,
                               kernel_size=(3, 3),
                               stride=self.stride,
                               padding=(1, 1))
        self.conv3 = nn.Conv2d(self.mid_channel, self.out_channel,
                               kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4, "input should be in (N*T, C, H, W)"
        N, C, H, W = x.size()
        T = self.time_length
        N = int(N / T)
        C = C // 16

        x = self.conv1(x)

        x1 = x.view((N, T, C, H, W))[:, range(T-1),  :, :, :].view((N * (T-1), C, H, W))
        x2 = x.view((N, T, C, H, W))[:, range(1, T), :, :, :].view((N * (T-1), C, H, W))

        # Notice: I've found that the paper didn't cover that
        #  once we replace the original 3x3 2D conv with CMM
        #  how can we implement the downsampling
        #  here I choose to downsample our data in the second conv (3x3 2D)
        #  in the meanwhile, via a constant ones tensor and another 2D conv
        #  the "raw" feature will be downsampled too
        if self.stride > 1:
            weight = torch.ones((self.mid_channel, self.mid_channel, 3, 3)).cuda()
            x1 = F.conv2d(x1, weight, stride=self.stride, padding=1)
            H //= 2
            W //= 2
        x = self.conv2(x2) - x1
        x = self.conv3(x)

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

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(planes)

        self.cstm = CSTM(planes, stride=stride, time_length=time_length)
        self.cmm = CMM(planes, stride=stride, time_length=time_length)

        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

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

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = self.relu(x)

        return x
