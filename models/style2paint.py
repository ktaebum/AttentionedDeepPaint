"""
Style2Paint model architecture implementation
"""

import torch
import torch.nn as nn


class StylePaintDiscriminator(nn.Module):
    def __init__(self, sigmoid=True):
        super(StylePaintDiscriminator, self).__init__()
        self.dim = 16
        self.num_class = 4096
        self.sigmoid = sigmoid

        layers = []
        # self.dim x 256 x 256
        layers.append(self._single_conv_block(3, self.dim, False))

        # self.dim * 2 x 128 x 128
        layers.append(self._single_conv_block(self.dim, self.dim * 2))

        # self.dim * 4 x 64 x 64
        layers.append(self._single_conv_block(self.dim * 2, self.dim * 4))

        # self.dim * 8 x 32 x 32
        layers.append(self._single_conv_block(self.dim * 4, self.dim * 8))

        # self.dim * 16 x 16 x16
        layers.append(self._single_conv_block(self.dim * 8, self.dim * 16))

        # self.dim * 16 x 8 x 8
        layers.append(self._single_conv_block(self.dim * 16, self.dim * 16))

        # self.dim * 32 x 4 x 4
        layers.append(
            self._single_conv_block(self.dim * 16, self.dim * 32, False))

        self.convs = nn.Sequential(*layers)
        self.linear = nn.Linear(self.dim * 32 * 4 * 4, self.num_class)

        for module in self.convs.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        image = self.convs(image)
        image = image.reshape(image.shape[0], -1)
        image = self.linear(image)
        if self.sigmoid:
            image = torch.sigmoid(image)
        return image

    def _single_conv_block(self,
                           in_channels,
                           out_channels,
                           norm=True,
                           relu=True):
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if relu:
            layers.append(nn.LeakyReLU(0.2, True))

        return nn.Sequential(*layers)


class StylePaintGenerator(nn.Module):
    def __init__(self, bias=True, dropout=0.5, norm='batch'):
        super(StylePaintGenerator, self).__init__()

        self.bias = bias
        self.dropout = dropout
        self.dim = 16

        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            self.norm = None

        self.additional_fc2 = nn.Sequential(nn.Linear(4096, 2048))

        self.down_sampler = self._build_downsampler()
        self.up_sampler = self._build_upsampler()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.dim * 32, 2048, 4, 2, 1, bias=self.bias),
            self.norm(2048) if self.norm is not None else nn.Sequential(),
        )

        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.dim, 7, 1, 3, bias=self.bias),
            self.norm(self.dim), nn.LeakyReLU(0.2, True))
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.dim * 2, 3, 7, 1, 3, bias=self.bias), self.norm(3),
            nn.Tanh())

        self.guide_decoder1 = self._build_guide_decoder()
        self.guide_decoder2 = self._build_guide_decoder()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image, style):
        style = self.additional_fc2(style)

        skip_connections = []

        image = self.first_layer(image)
        skip_connections.append(image)

        for layer in self.down_sampler:
            image = layer(image)
            skip_connections.append(image)

        guide1 = self.guide_decoder1(image)

        image = self.bottleneck(image)

        style = style.unsqueeze(-1).unsqueeze(-1)
        image = image + style
        torch.relu(image)

        skip_connections = list(reversed(skip_connections))

        # now upsample

        # propagate the first upsample layer
        image = self.up_sampler[0](image)
        guide2 = self.guide_decoder2(image)
        image = torch.cat([image, skip_connections[0]], 1)

        for layer, connection in zip(self.up_sampler[1:],
                                     skip_connections[1:]):
            image = layer(image)
            image = torch.cat([image, connection], 1)

        image = self.last_layer(image)

        return image, guide1, guide2

    def _build_guide_decoder(self):
        def guide_block(in_channels, out_channels):
            layers = []

            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 4, 2, 1, bias=self.bias))
            if self.norm is not None:
                layers.append(self.norm(out_channels))

            layers.append(nn.ReLU(True))

            return nn.Sequential(*layers)

        layers = []
        layers.append(guide_block(self.dim * 32, self.dim * 16))
        layers.append(guide_block(self.dim * 16, self.dim * 8))
        layers.append(guide_block(self.dim * 8, self.dim * 4))
        layers.append(guide_block(self.dim * 4, self.dim * 2))
        layers.append(guide_block(self.dim * 2, self.dim * 1))
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.dim * 1, 3, 7, 1, 3, bias=self.bias),
                self.norm(3) if self.norm is not None else nn.Sequential(),
                nn.Tanh()))

        return nn.Sequential(*layers)

    def _build_upsampler(self):
        """
        Reversed operation of downsampler
        """

        layers = nn.ModuleList()
        layers.append(
            SingleLayerUpSampleBlock(2048, self.dim * 32, self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 32 * 2, self.dim * 16,
                                     self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 16 * 2, self.dim * 8,
                                     self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 8 * 2, self.dim * 4,
                                     self.bias))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 4 * 2, self.dim * 2,
                                     self.bias))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 2 * 2, self.dim * 1,
                                     self.bias))
        """
        # self.dim * 32 x 16 x 16
        layers.append(
            TwoLayerUpSampleBlock(2048, self.dim * 32, True, self.norm, False,
                                  self.bias))

        # self.dim * 16 x 32 x 32
        layers.append(
            TwoLayerUpSampleBlock(self.dim * 32 * 2, self.dim * 16, True,
                                  self.norm, False, self.bias))

        # self.dim * 8 x 64 x 64
        layers.append(
            TwoLayerUpSampleBlock(self.dim * 16 * 2, self.dim * 8, True,
                                  self.norm, False, self.bias))

        # self.dim * 4 x 128 x 128
        layers.append(
            TwoLayerUpSampleBlock(self.dim * 8 * 2, self.dim * 4, True,
                                  self.norm, False, self.bias))

        # self.dim * 2 x 256 x 256
        layers.append(
            TwoLayerUpSampleBlock(self.dim * 4 * 2, self.dim * 2, True,
                                  self.norm, False, self.bias))

        # self.dim x 512 x 512
        layers.append(
            TwoLayerUpSampleBlock(self.dim * 2 * 2, self.dim, True, self.norm,
                                  False, self.bias))
        """

        return layers

    def _build_downsampler(self):
        """
        Although original paper aims to colorize 256 x 256

        But our goal is 512 x 512
        """
        layers = nn.ModuleList()

        layers.append(
            SingleLayerDownSampleBlock(self.dim, self.dim * 2, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 2, self.dim * 4, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 4, self.dim * 8, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 8, self.dim * 16, self.bias))
        layers.append(
            SingleLayerDownSampleBlock(self.dim * 16, self.dim * 32,
                                       self.bias))
        """
        # self.dim x 512 x 512
        layers.append(
            TwoLayerDownSampleBlock(3, self.dim, False, self.norm, self.bias))

        # self.dim * 2 x 256 x 256
        layers.append(
            TwoLayerDownSampleBlock(self.dim, self.dim * 2, True, self.norm,
                                    self.bias))

        # self.dim * 4 x 128 x 128
        layers.append(
            TwoLayerDownSampleBlock(self.dim * 2, self.dim * 4, True,
                                    self.norm, self.bias))

        # self.dim * 8 x 64 x 64
        layers.append(
            TwoLayerDownSampleBlock(self.dim * 4, self.dim * 8, True,
                                    self.norm, self.bias))

        # self.dim * 16 x 32 x 32
        layers.append(
            TwoLayerDownSampleBlock(self.dim * 8, self.dim * 16, True,
                                    self.norm, self.bias))

        # self.dim * 32 x 16 x 16
        layers.append(
            TwoLayerDownSampleBlock(self.dim * 16, self.dim * 32, True,
                                    self.norm, self.bias))
        """

        return layers


class SingleLayerDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SingleLayerDownSampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, feature):
        feature = self.conv(feature)
        feature = self.norm(feature)
        feature = self.activation(feature)

        return feature


class SingleLayerUpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=False):
        super(SingleLayerUpSampleBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, 4, 2, 1, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5, True) if dropout else nn.Sequential()
        self.activation = nn.ReLU(True)

    def forward(self, feature):
        feature = self.conv(feature)
        feature = self.norm(feature)
        feature = self.dropout(feature)
        feature = self.activation(feature)

        return feature


class TwoLayerDownSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample=True,
                 norm=None,
                 bias=True):
        super(TwoLayerDownSampleBlock, self).__init__()

        if downsample:
            self.first_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.first_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        self.lrelu1 = nn.LeakyReLU(0.2, True)

        self.second_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)
        self.lrelu2 = nn.LeakyReLU(0.2, True)
        if not norm:
            self.norm1 = norm(out_channels)
            self.norm2 = norm(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, feature):
        first_feature = self.first_conv(feature)
        if self.norm1:
            first_feature = self.norm1(first_feature)
        first_feature = self.lrelu1(first_feature)

        second_feature = self.second_conv(first_feature)
        if self.norm2:
            second_feature = self.norm2(second_feature)
        second_feature = self.lrelu2(second_feature)

        return first_feature, second_feature


class TwoLayerUpSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample=True,
                 norm=None,
                 is_last=False,
                 bias=True):
        super(TwoLayerUpSampleBlock, self).__init__()

        self.is_last = is_last

        if upsample:
            self.first_conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.first_conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        self.relu1 = nn.ReLU(True)

        self.second_conv = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)
        self.relu2 = nn.ReLU(True)
        if not norm:
            self.norm1 = norm(out_channels)
            self.norm2 = norm(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None

        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, feature):
        first_feature = self.first_conv(feature)
        if self.norm1:
            first_feature = self.norm1(first_feature)
        first_feature = self.relu1(first_feature)

        second_feature = self.second_conv(first_feature)
        if self.norm2:
            second_feature = self.norm2(second_feature)

        if self.is_last:
            second_feature = torch.tanh(second_feature)
        else:
            second_feature = self.relu2(second_feature)

        return first_feature, second_feature
