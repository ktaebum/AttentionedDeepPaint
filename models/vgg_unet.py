"""
Combined VGG-19 with BatchNorm and Normal UNet (pix2pix model) as generator

Based reference on style2paints paper
"""

import torch
import torch.nn as nn

from torchvision import models


class VggUnet(nn.Module):
    def __init__(self,
                 resolution=256,
                 dim=64,
                 bias=True,
                 norm='batch',
                 dropout=0.5):
        super(VggUnet, self).__init__()

        self.resolution = resolution
        self.dim = dim
        self.bias = bias
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            raise ValueError('Invalid Normalization')

        self.cfg = [
            3, dim, dim * 2, dim * 4, dim * 8, dim * 8, dim * 8, dim * 16
        ]

        if self.resolution == 256:
            self.cfg += [dim * 32]
        elif self.resolution == 512:
            self.cfg += [dim * 16, dim * 32]
        else:
            raise ValueError('Invalid Resolution')

        # get pretrained vgg net
        pretrained_vgg = list(models.vgg19_bn(True).children())
        self.vgg_conv = pretrained_vgg[0]
        self.vgg_fc1 = pretrained_vgg[1][0]

        # turn off gradient
        for param in self.vgg_conv.parameters():
            param.requires_grad = False

        for param in self.vgg_fc1.parameters():
            param.requires_grad = False

        self.vgg_fc2 = nn.Linear(4096, 2048, bias=bias)

        self.down_sampler = self._build_downsampler()
        self.up_sampler = self._build_upsampler()

        self.guide_decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=self.bias)
            if self.resolution == 512 else nn.Sequential(),
            nn.ReLU(True) if self.resolution == 512 else nn.Sequential(),
            # 16 x 16
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 128 x 128
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 256 x 256
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=self.bias),
            nn.Tanh())

        self.guide_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=self.bias)
            if self.resolution == 512 else nn.Sequential(),
            nn.ReLU(True) if self.resolution == 512 else nn.Sequential(),
            # 16 x 16
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 128 x 128
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=self.bias),
            nn.ReLU(True),
            # 256 x 256
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=self.bias),
            nn.Tanh())

    def forward(self, image, reference):

        # calculate for style reference
        reference = self.vgg_conv(reference)
        reference = reference.reshape(reference.shape[0], -1)
        reference = self.vgg_fc1(reference)
        reference = self.vgg_fc2(reference)

        skip_connections = []

        guide1 = None
        guide2 = None

        for i, layer in enumerate(self.down_sampler, 1):
            image = layer(image)
            if image.shape[-1] == 8:
                # extract for guide decoder1
                guide1 = self.guide_decoder1(image)
            if i < len(self.down_sampler):
                skip_connections.append(image)

        skip_connections = list(reversed(skip_connections))

        # concat with style reference
        reference = reference.unsqueeze(-1).unsqueeze(-1)
        image = torch.cat([image, reference], 1)

        for i, (connection, layer) in enumerate(
                zip(skip_connections, self.up_sampler[:-1]), 1):
            image = layer(image)
            if i <= 4:
                image = self.dropout(image)
            image = torch.relu(image)

            if image.shape[-1] == 8:
                # extract for guide decoder 2
                guide2 = self.guide_decoder2(image)

            # concat with skip connection
            image = torch.cat([image, connection], 1)

        image = self.up_sampler[-1](image)
        image = torch.tanh(image)

        return image, guide1, guide2

    def _build_downsampler(self):
        """
        Build Down Sampler

        For the last layer, since it produces (batch_size, self.dim * 32, 1, 1)
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
                    padding=1,
                    bias=self.bias),
                self.norm(self.cfg[i + 1])
                if i < len(self.cfg) - 2 else nn.Sequential(),
                nn.LeakyReLU(0.2, inplace=True),
            )
            layers.append(block)

        return layers

    def _build_upsampler(self):
        """
        Build Up Sampler

        For the last layer, do not use normalization (direct tanh).
        Also, since the dropout could be at the early step in upsampling,
        do not append relu directly in layer block.
        """
        layers = nn.ModuleList()

        for i in reversed(range(1, len(self.cfg))):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.cfg[i] * 2,
                    #  (1 if i == len(self.cfg) - 1 else 2),
                    out_channels=self.cfg[i - 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=self.bias),
                self.norm(self.cfg[i - 1]) if i > 1 else nn.Sequential())

            layers.append(block)

        return layers
