"""
Residual Block Module for generator

Referred Cycle GAN github
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
    blob/master/models/networks.py
"""

import torch.nn as nn


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
