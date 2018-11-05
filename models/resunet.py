"""
Unet generator using ResBlock

Almost same with VggUnet, but build seperated script for clearance
"""

import torch
import torch.nn as nn

from torchvision import models

from models import HalfResBlock, ResBlock
from models.style2paint import SingleLayerDownSampleBlock
from models.style2paint import SingleLayerUpSampleBlock


class ResidualUnet(nn.Module):
    def __init__(self, bias=True, norm='batch', dropout=0.5):
        super(ResidualUnet, self).__init__()
        self.dim = 64
        self.bias = bias
        self.guide_resolution = 8
        self.bridge_channel = 2048
        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            self.norm = None

        self.downsampler = self._build_downsampler()

        self.to_bridge = nn.Sequential(
            nn.Conv2d(
                self.dim * 16, self.bridge_channel, 4, 2, 1, bias=self.bias),
            self.norm(self.bridge_channel)
            if self.norm is not None else nn.Sequential())
        self.upsampler = self._build_upsampler()

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.dim * 2, 3, 4, 2, 1, bias=self.bias),
            nn.Tanh())

        self.guide_decoder1 = self._guide_decoder(True)
        self.guide_decoder2 = self._guide_decoder(False)

        # for feature embedding from vggnet
        if self.bridge_channel != 4096:
            self.feature_embed = nn.Sequential(
                nn.Linear(4096, self.bridge_channel, bias=self.bias))
        else:
            self.feature_embed = None

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image, style):
        if self.feature_embed is not None:
            style = self.feature_embed(style)
        style = style.unsqueeze(-1).unsqueeze(-1)

        skip_connections = []

        # run downsampling
        for layer in self.downsampler:
            connection, image = layer(image)
            skip_connections.append(connection)
            if image.shape[-1] == self.guide_resolution:
                guide1 = self.guide_decoder1(image)

        # mid bridge
        image = self.to_bridge(image)
        image = image + style
        image = torch.relu(image)

        # run upsampling
        for connection, layer in zip(
                reversed(skip_connections), self.upsampler):
            image = layer(image, connection)
            if image.shape[-1] == self.guide_resolution:
                guide2 = self.guide_decoder2(image)

        image = self.final_layer(image)

        return image, guide1, guide2

    def _guide_decoder(self, is_first):
        """
        Assume input is 8 x 8
        """

        def guide_block(in_channels, out_channels):
            blocks = []
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 4, 2, 1, bias=self.bias))
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(True))
            return nn.Sequential(*blocks)

        in_channels = self.dim * 8 if is_first else self.dim * 16
        layers = []
        # 16 x 16
        layers.append(guide_block(in_channels, self.dim * 8))
        # 32 x 32
        layers.append(guide_block(self.dim * 8, self.dim * 8))
        # 64 x 64
        layers.append(guide_block(self.dim * 8, self.dim * 4))
        # 128 x 128
        layers.append(guide_block(self.dim * 4, self.dim * 2))
        # 256 x 256
        layers.append(guide_block(self.dim * 2, self.dim * 1))
        # 512 x 512
        layers.append(nn.ConvTranspose2d(self.dim, 3, 4, 2, 1))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _build_downsampler(self):
        layers = nn.ModuleList()
        layers.append(SingleLayerDownSampleBlock(3, self.dim, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim, self.dim * 2, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 2, self.dim * 4, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 4, self.dim * 8, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 8, self.dim * 8, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 8, self.dim * 8, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 8, self.dim * 16, self.bias))

        return layers

    def _build_upsampler(self, skip_connection=True):
        """
        skip connected upsampler
        """
        layers = nn.ModuleList()

        # 4 x 4
        layers.append(
            SingleLayerUpSampleBlock(self.bridge_channel, self.dim * 16,
                                     self.bias, True))

        # 8 x 8
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 16 * 2, self.dim * 8,
                                     self.bias, True))

        # 16 x 16
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 8 * 2, self.dim * 8, self.bias,
                                     True))

        # 32 x 32
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 8 * 2, self.dim * 8, self.bias,
                                     True))

        # 64 x 64
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 8 * 2, self.dim * 4, self.bias,
                                     False))

        # 128 x 128
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 4 * 2, self.dim * 2, self.bias,
                                     False))

        # 256 x 256
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 2 * 2, self.dim, self.bias,
                                     False))
        return layers


class ResUnet(nn.Module):
    def __init__(self, resblock=True, bias=True, norm='batch', dropout=0.5):
        super(ResUnet, self).__init__()

        self.dim = 64
        self.bias = bias
        self.norm = norm
        self.resblock = resblock
        self.guide_resolution = 16

        if norm == 'batch':
            self.norm_block = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm_block = nn.InstanceNorm2d
        else:
            raise ValueError('Invalid Normalization')

        self.dropout = dropout
        self.dropout_block = nn.Dropout2d(
            dropout) if dropout > 0 else nn.Sequential()

        self.cfg = [
            3, self.dim, self.dim * 2, self.dim * 4, self.dim * 4,
            self.dim * 8, self.dim * 8, self.dim * 8, self.dim * 16,
            self.dim * 32
        ]

        vgg = models.vgg19_bn(True)
        self.vgg_features = vgg.features
        self.vgg_fc1 = vgg.classifier[0]

        self.feature_embed = nn.Linear(4096, self.dim * 32)

        for parameter in self.vgg_features.parameters():
            parameter.requires_grad = False
        for parameter in self.vgg_fc1.parameters():
            parameter.requires_grad = False

        self.down_sampler = self._build_downsampler(self.resblock)
        self.up_sampler = self._build_upsampler(self.resblock)

        self.guide_decoder1 = self._build_guide_decoder()
        self.guide_decoder2 = self._build_guide_decoder()

        self._initialize_weight()

    def forward(self, image, style):
        with torch.no_grad():
            style = self.vgg_features(style)
            style = style.reshape(style.shape[0], -1)
            style = self.vgg_fc1(style)

        style = self.feature_embed(style)

        skip_connections = []

        guide1 = None
        guide2 = None

        for i, layer in enumerate(self.down_sampler, 1):
            image = layer(image)
            if image.shape[-1] == self.guide_resolution:
                guide1 = self.guide_decoder1(image)

            if i < len(self.down_sampler):
                skip_connections.append(image)

        skip_connections = list(reversed(skip_connections))

        style = style.unsqueeze(-1).unsqueeze(-1)
        image = image + style

        for i, (connection, layer) in enumerate(
                zip(skip_connections, self.up_sampler[:-1]), 1):
            image = layer(image)
            if i <= 3:
                image = self.dropout_block(image)
            image = torch.relu(image)

            if image.shape[-1] == self.guide_resolution:
                guide2 = self.guide_decoder2(image)

            image = torch.cat([image, connection], 1)

        image = self.up_sampler[-1](image)
        image = torch.tanh(image)

        return image, guide1, guide2

    def _initialize_weight(self):
        initial_targets = [
            self.down_sampler, self.up_sampler, self.guide_decoder1,
            self.guide_decoder2
        ]

        for target in initial_targets:
            for module in target.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, 0, 0.02)
                elif isinstance(module, nn.ConvTranspose2d):
                    nn.init.normal_(module.weight, 0, 0.02)

    def _build_guide_decoder(self):
        """
        Build guide decoder

        From 512 x 16 x 16 feature, make 3 x 512 x 512 image

        512 x 16 x 16
        256 x 32 x 32
        256 x 64 x 64
        128 x 128 x 128
        64 x 256 x 256
        3 x 512 x 512

        From 2048 x 2 x 2 feature, make 3 x 512 x 512 image

        2048 x 2 x 2
        1024 x 4 x 4
        512 x 8 x 8
        512 x 16 x 16
        256 x 32 x 32
        256 x 64 x 64
        128 x 128 x 128
        64 x 256 x 256
        3 x 512 x 512
        """

        def guide_block(in_channels, out_channels, norm=True, relu=True):
            block = [
                nn.ConvTranspose2d(
                    in_channels, out_channels, 4, 2, 1, bias=self.bias)
            ]

            if norm:
                block.append(self.norm_block(out_channels))
            if relu:
                block.append(nn.ReLU(True))

            return block

        guide_decoder = nn.Sequential(
            #  *guide_block(self.dim * 32, self.dim * 16),
            #  *guide_block(self.dim * 16, self.dim * 8),
            #  *guide_block(self.dim * 8, self.dim * 8),
            *guide_block(self.dim * 8, self.dim * 4),
            *guide_block(self.dim * 4, self.dim * 4),
            *guide_block(self.dim * 4, self.dim * 2),
            *guide_block(self.dim * 2, self.dim),
            *guide_block(self.dim, 3, norm=False, relu=False),
            nn.Tanh())

        return guide_decoder

    def _build_downsampler(self, resblock=True):
        """
        Build Downsampler
        """
        layers = nn.ModuleList()

        for i in range(len(self.cfg) - 1):
            if resblock:
                block = nn.Sequential(
                    HalfResBlock(
                        self.cfg[i],
                        self.cfg[i + 1],
                        norm=(i < len(self.cfg) - 2),
                        dropout=self.dropout,
                        bias=self.bias,
                        mode='down'), nn.LeakyReLU(0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.cfg[i],
                        out_channels=self.cfg[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=self.bias),
                    self.norm_block(self.cfg[i + 1])
                    if i < len(self.cfg) - 2 else nn.Sequential(),
                    nn.LeakyReLU(0.2, inplace=True))

            layers.append(block)

        return layers

    def _build_upsampler(self, resblock=True):
        """
        Build Upsampler
        """

        layers = nn.ModuleList()

        for i in reversed(range(1, len(self.cfg))):
            if resblock:
                block = nn.Sequential(
                    HalfResBlock(
                        self.cfg[i] * (1 if i == len(self.cfg) - 1 else 2),
                        self.cfg[i - 1],
                        norm=(i > 1),
                        dropout=self.dropout,
                        bias=self.bias,
                        mode='up'))
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.cfg[i] *
                        (1 if i == len(self.cfg) - 1 else 2),
                        out_channels=self.cfg[i - 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=self.bias),
                    self.norm_block(self.cfg[i - 1])
                    if i > 1 else nn.Sequential())

            layers.append(block)

        return layers
