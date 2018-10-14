"""
Residual Block Module for generator

Referred Cycle GAN github
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
    blob/master/models/networks.py
"""

import torch.nn as nn


class ShortCutMap(nn.Module):
    """
    Shortcut in channel, size changing residual block.
    This is the simplest version, which use Conv2d in downsampling
    and ConvTranspose2d in upsampling
    """

    def __init__(self, in_channels, out_channels, mode, stride=2):
        super(ShortCutMap, self).__init__()

        if mode == 'down':
            self.sample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False)
        elif mode == 'up':
            self.sample = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                output_padding=1,
                bias=False)
        else:
            raise ValueError('Invalid Mode')

    def forward(self, x):
        return self.sample(x)


class HalfResBlock(nn.Module):
    """
    Half downsampling residual block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 mode='down',
                 dropout=0.5,
                 bias=True):
        super(HalfResBlock, self).__init__()

        self.mode = mode
        self.shortcut = ShortCutMap(in_channels, out_channels, mode)
        if mode == 'down':
            self.conv = nn.Conv2d
        elif mode == 'up':
            self.conv = nn.ConvTranspose2d
        else:
            raise ValueError('Invalid mode')

        self.conv1 = self.conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            padding=1,
            stride=2,
            bias=bias)

        if norm:
            self.norm1 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Sequential()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d(
            dropout) if dropout > 0 else nn.Sequential()

        self.conv2 = self.conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=bias)

        if norm:
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm2 = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)

        return out + self.shortcut(x)


class ResBlock(nn.Module):
    def __init__(self,
                 channels,
                 pad_type='normal',
                 norm='instance',
                 dropout=0.5,
                 bias=True):
        """
        Constructor for residual block.
        Each block has two convolutional layers

        @param channels: input & output channels
        @param pad_type: padding type (default as 'normal')
            Reflection ('reflect'), Replication ('replicate') available
        @param norm: normalization of conv layer (instance or batch)
        @param dropout: dropout probability
        @param bias: whether use bias or not
        """
        super(ResBlock, self).__init__()

        if norm == 'instance':
            self.norm = nn.InstanceNorm2d
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d
        else:
            raise ValueError('Invalid normalization layer %s' % str(norm))

        self.pad_type = pad_type
        self.block = []

        # first conv
        padding = self._add_padding_layer()
        self.block.append(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                bias=bias,
            ))
        self.block.append(self.norm(channels))
        self.block.append(nn.ReLU(True))

        # add dropout
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        # second conv
        padding = self._add_padding_layer()
        self.block.append(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                bias=bias,
            ))
        self.block.append(self.norm(channels))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        out = self.block(x)
        return out + x

    def _add_padding_layer(self):
        """
        Add padding layer to self.block
        """
        padding = 0
        if self.pad_type == 'normal':
            padding = 1
        elif self.pad_type == 'reflect':
            self.block.append(nn.ReflectionPad2d(1))
        elif self.pad_type == 'replicate':
            self.block.append(nn.ReplicationPad2d(1))
        else:
            raise ValueError('Invalid padding type %s' % str(self.pad_type))

        return padding
