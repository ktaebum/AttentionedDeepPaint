"""
Style2Paint model architecture implementation
"""

import torch
import torch.nn as nn


class StylePaintDiscriminator(nn.Module):
    def __init__(self, sigmoid=True):
        super(StylePaintDiscriminator, self).__init__()
        self.dim = 64
        self.sigmoid = sigmoid

        layers = []
        # 256 x 256
        layers.append(self._single_conv_block(6, self.dim, False))

        # 128 x 128
        layers.append(self._single_conv_block(self.dim, self.dim * 2))

        # 64 x 64
        layers.append(self._single_conv_block(self.dim * 2, self.dim * 4))

        # 32 x 32
        layers.append(self._single_conv_block(self.dim * 4, self.dim * 8))

        # 16 x16
        layers.append(self._single_conv_block(self.dim * 8, self.dim * 8))

        # 8 x 8
        layers.append(self._single_conv_block(self.dim * 8, self.dim * 8))

        # 4 x 4
        layers.append(self._single_conv_block(self.dim * 8, self.dim * 8))

        # 2 x 2
        layers.append(self._single_conv_block(self.dim * 8, self.dim * 8))

        # 1 x 1
        layers.append(
            nn.Conv2d(
                self.dim * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False))

        self.convs = nn.Sequential(*layers)

        for module in self.convs.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        image = self.convs(image)
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
        self.dim = 32
        self.bottleneck_channel = 2048

        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)

        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            self.norm = None

        self.first_layer = nn.Sequential(
            nn.Conv2d(3, self.dim, 7, 1, 3, bias=self.bias),
            self.norm(self.dim) if self.norm is not None else nn.Sequential())

        self.down_sampler = self._build_downsampler()

        # from down_sampler to mid_layer (bottleneck)
        self.to_bottleneck = nn.Sequential(
            nn.Conv2d(
                self.dim * 32,
                self.bottleneck_channel,
                4,
                2,
                1,
                bias=self.bias),
            self.norm(self.bottleneck_channel)
            if self.norm is not None else nn.Sequential(),
        )
        if self.bottleneck_channel < 4096:
            self.embedding = nn.Linear(4096, self.bottleneck_channel)
        else:
            self.embedding = None

        self.up_sampler = self._build_upsampler()

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.dim * 2, 3, 7, 1, 3, bias=self.bias), nn.Tanh())

        # additional loss
        self.guide_decoder1 = self._build_guide_decoder(True)
        self.guide_decoder2 = self._build_guide_decoder(False)

        # initialize
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image, style):
        # make dimension (1, 4096, 1, 1)
        if self.embedding is not None:
            style = self.embedding(style)
        style = style.unsqueeze(-1).unsqueeze(-1)

        skip_connections = []

        # first layer
        image = self.first_layer(image)
        skip_connections.append(image)
        image = self.lrelu(image)

        # query downsampler
        for layer in self.down_sampler:
            connection, image = layer(image)
            skip_connections.append(connection)
        guide1 = self.guide_decoder1(image)

        skip_connections = list(reversed(skip_connections))
        # to bottleneck
        image = self.to_bottleneck(image)
        # add style
        image = image + style
        image = self.relu(image)

        for i, (layer, connection) in enumerate(
                zip(self.up_sampler, skip_connections)):
            image = layer(image, connection)
            if i == 0:
                guide2 = self.guide_decoder2(image)

        # final layer
        image = self.last_layer(image)

        return image, guide1, guide2

    def _build_guide_decoder(self, is_first=True):
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
        first_in_channels = self.dim * 32 if is_first else self.dim * 64
        layers.append(guide_block(first_in_channels, self.dim * 32))
        layers.append(guide_block(self.dim * 32, self.dim * 16))
        layers.append(guide_block(self.dim * 16, self.dim * 8))
        layers.append(guide_block(self.dim * 8, self.dim * 4))
        layers.append(guide_block(self.dim * 4, self.dim * 2))
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.dim * 2, 3, 7, 1, 3, bias=self.bias),
                nn.Tanh()))

        return nn.Sequential(*layers)

    def _build_upsampler(self):
        """
        Reversed operation of downsampler
        """

        layers = nn.ModuleList()
        layers.append(
            SingleLayerUpSampleBlock(self.bottleneck_channel, self.dim * 32,
                                     self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 32 + self.dim * 32,
                                     self.dim * 16, self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 16 + self.dim * 16,
                                     self.dim * 8, self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 8 + self.dim * 8, self.dim * 4,
                                     self.bias, True))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 4 + self.dim * 4, self.dim * 2,
                                     self.bias))
        layers.append(
            SingleLayerUpSampleBlock(self.dim * 2 + self.dim * 2, self.dim * 1,
                                     self.bias))
        return layers

    def _build_downsampler(self):
        """
        downsampler
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
        return layers


class SingleLayerDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SingleLayerDownSampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, feature):
        feature = self.conv(feature)
        normed = self.norm(feature)
        feature = self.activation(normed)

        return normed, feature


class SingleLayerUpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=False):
        super(SingleLayerUpSampleBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, 4, 2, 1, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5, True) if dropout else nn.Sequential()
        self.activation = nn.ReLU(True)

    def forward(self, feature, connection):
        feature = self.conv(feature)
        feature = self.norm(feature)
        feature = self.dropout(feature)
        feature = torch.cat([feature, connection], 1)
        feature = self.activation(feature)

        return feature
