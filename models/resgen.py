"""
Taebum
Residual generator architecture
"""

import torch
import torch.nn as nn

from models import ResBlock


class ResidualGenerator(nn.Module):
    def __init__(self,
                 resolution=512,
                 dim=64,
                 res_blocks=8,
                 bias=True,
                 norm='instance',
                 pad_type='reflect',
                 dropout=0.5):
        super(ResidualGenerator, self).__init__()

        self.resolution = resolution
        self.dim = dim
        self.bias = bias
        self.pad_type = pad_type
        self.norm_type = norm
        self.dropout = dropout

        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            raise ValueError('Invalid Normalization')

        self.down_cfg = [dim, dim * 2, dim * 4, dim * 8]
        self.res_cfg = [dim * 8] * res_blocks
        self.up_cfg = [dim * 8, dim * 4, dim * 2, dim]

        self.conv_first = nn.Conv2d(
            6,
            self.dim,
            kernel_size=7,
            padding=3,
            bias=bias,
        )

        self.down_sampler = self._build_downsampler()
        self.residual_layer = self._build_residual_layers()
        self.up_sampler = self._build_upsampler()

        self.conv_last = nn.Conv2d(
            dim,
            3,
            kernel_size=7,
            padding=3,
            bias=bias,
        )

        self._initialize_weights()

    def forward(self, image, style):
        image = torch.cat([image, style], 1)

        image = self.conv_first(image)
        image = self.down_sampler(image)
        image = self.residual_layer(image)
        image = self.up_sampler(image)
        image = self.conv_last(image)

        return torch.tanh(image)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def _build_downsampler(self):
        """
        Down sampler for residual generator
        """

        layers = nn.ModuleList()

        for i in range(len(self.down_cfg) - 1):
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.down_cfg[i],
                    out_channels=self.down_cfg[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=self.bias),
                self.norm(self.down_cfg[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            )

            layers.append(block)

        return nn.Sequential(*layers)

    def _build_residual_layers(self):
        """
        Residual layers in generator
        """

        layers = nn.ModuleList()

        for cfg in self.res_cfg:
            layers.append(
                ResBlock(
                    cfg,
                    pad_type=self.pad_type,
                    norm=self.norm_type,
                    dropout=self.dropout,
                    bias=self.bias))

        return nn.Sequential(*layers)

    def _build_upsampler(self):
        layers = nn.ModuleList()

        for i in range(len(self.up_cfg) - 1):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.up_cfg[i],
                    out_channels=self.up_cfg[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=self.bias), self.norm(self.up_cfg[i + 1]),
                nn.ReLU(True))

            layers.append(block)

        return nn.Sequential(*layers)
