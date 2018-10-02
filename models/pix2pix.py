"""
pix2pix generator model

Assume that we use pix2pix generator for 512x512 resolution image

Model architecture becomes

input image (3 x 512 x 512)

Downsample1 -> (64 x 256 x 256)
Downsample2 -> (128 x 128 x 128)
Downsample3 -> (256 x 64 x 64)
Downsample4 -> (256 x 32 x 32)
Downsample5 -> (512 x 16 x 16)
Downsample6 -> (512 x 8 x 8)
Downsample7 -> (512 x 4 x 4)
Downsample8 -> (512 x 2 x 2)
Downsample9 -> (1024 x 1 x 1)

Upsample recover original shape (3 x 512 x 512)

Skip connection used
"""

import torch
import torch.nn as nn


class Pix2PixGenerator(nn.Module):
    def __init__(self, dim=64, dropout=0.5, norm='instance'):
        super(Pix2PixGenerator, self).__init__()

        self.dim = dim

        # channel configuration
        # use forward in downsample, reversed in upsample
        #  self.cfg = [
        #  3, dim, dim * 2, dim * 4, dim * 4, dim * 8, dim * 8, dim * 8,
        #  dim * 8, dim * 16
        #  ]
        self.cfg = [
            3, dim, dim * 2, dim * 4, dim * 8, dim * 8, dim * 8, dim * 8,
            dim * 8, dim * 16
        ]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
        self.norm = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
        self.down_sampler = self._down_sample()
        self.up_sampler = self._up_sample()

        # initialize weight
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        """
        Forward pass for input image

        @param image: input image with shape [Batch Size, 3, 512, 512]

        returns: Generated image with shape [Batch Size, 3, 512, 512]
        """

        # list for skip connections
        skip_connections = []

        for i, layer in enumerate(self.down_sampler, 1):
            image = layer(image)
            if i < len(self.down_sampler):
                # append feature to skip connections
                # do not append the last layer's feature
                skip_connections.append(image)

        # reverse skip connection
        skip_connections = list(reversed(skip_connections))

        for i, (connection, layer) in enumerate(
                zip(skip_connections, self.up_sampler[:-1]), 1):
            image = layer(image)
            if i <= 3:
                # dropout!
                image = self.dropout(image)
            image = torch.relu(image)

            # concat with skip connection
            image = torch.cat([image, connection], 1)

        # query for last layer
        image = self.up_sampler[-1](image)
        image = torch.tanh(image)

        return image

    def _down_sample(self):
        """
        Build Down Sampler

        For the last layer, since it produces (batch_size, self.dim * 16, 1, 1)
        , do not use normalization (it raises error!)
        """
        layers = nn.ModuleList()

        for i in range(len(self.cfg) - 1):
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.cfg[i],
                    out_channels=self.cfg[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1),
                self.norm(self.cfg[i + 1])
                if i < len(self.cfg) - 2 else nn.Sequential(),
                nn.LeakyReLU(0.2, inplace=True),
            )
            layers.append(block)

        return layers

    def _up_sample(self):
        """
        Build Up Sampler

        For the last layer, do not use normalization (direct tanh).
        Also, since the dropout could be at the early step in upsampling,
        do not append relu directly in layer block.
        """
        layers = nn.ModuleList()

        for i in reversed(range(1, len(self.cfg))):
            block = nn.Sequential(*[
                nn.ConvTranspose2d(
                    in_channels=self.cfg[i] *
                    (1 if i == len(self.cfg) - 1 else 2),
                    out_channels=self.cfg[i - 1],
                    kernel_size=4,
                    stride=2,
                    padding=1),
                self.norm(self.cfg[i - 1]) if i > 1 else nn.Sequential()
            ])

            layers.append(block)

        return layers
