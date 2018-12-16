"""
Improvement of DeepPaint With Residual Block
"""

import torch
import torch.nn as nn

from models.attention import AttentionBlock

Norm = nn.BatchNorm2d


class DeepUNetPaintGenerator(nn.Module):
    """
    Use Unet & SegNet & Residual Block feature
    """

    def __init__(self, bias=True):
        super(DeepUNetPaintGenerator, self).__init__()

        self.bias = bias
        self.dim = 64

        self.down_sampler = self._down_sample()
        self.up_sampler = self._up_sample()
        self.attentions = self._attention_blocks()

        self.first_layer = nn.Sequential(
            nn.Conv2d(15, self.dim, 3, 1, 1, bias=bias),
            Norm(self.dim),
        )
        self.gate_block = nn.Sequential(
            nn.Conv2d(
                self.dim * 8,
                self.dim * 8,
                kernel_size=1,
                stride=1,
                bias=self.bias), nn.BatchNorm2d(self.dim * 8))

        self.last_layer = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(self.dim, 3, 3, 1, 1, bias=bias),
            nn.Tanh(),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def test(self):
        x = torch.randn(1, 3, 512, 512)
        c = torch.randn(1, 12, 512, 512)
        print(self(x, c).shape)

    def forward(self, image, colors):
        cache = []
        image = torch.cat([image, colors], 1)

        image = self.first_layer(image)

        for i, layer in enumerate(self.down_sampler):
            image, connection, idx = layer(image)
            cache.append((connection, idx))

        cache = list(reversed(cache))
        gate = self.gate_block(image)
        attentions = []

        for i, (layer, attention, (connection, idx)) in enumerate(
                zip(self.up_sampler, self.attentions, cache)):
            connection, attr = attention(connection, gate)
            image = layer(image, connection, idx)
            attentions.append(attr)

        image = self.last_layer(image)

        return image, attentions

    def _attention_blocks(self):
        layers = nn.ModuleList()

        gate_channels = self.dim * 8

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 8, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 4, gate_channels, bias=self.bias))

        layers.append(
            AttentionBlock(self.dim * 2, gate_channels, bias=self.bias))

        return layers

    def _down_sample(self):
        layers = nn.ModuleList()

        # 256
        layers.append(DeepUNetDownSample(self.dim, self.dim * 2, self.bias))

        # 128
        layers.append(
            DeepUNetDownSample(self.dim * 2, self.dim * 4, self.bias))

        # 64
        layers.append(
            DeepUNetDownSample(self.dim * 4, self.dim * 8, self.bias))

        # 32
        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        # 16
        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        # 8
        layers.append(
            DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))

        return layers

    def _up_sample(self):
        layers = nn.ModuleList()
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepUNetUpSample(self.dim * 8 * 2, self.dim * 4, self.bias))
        layers.append(
            DeepUNetUpSample(self.dim * 4 * 2, self.dim * 2, self.bias))
        layers.append(DeepUNetUpSample(self.dim * 2 * 2, self.dim, self.bias))
        return layers


class DeepUNetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeepUNetDownSample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = Norm(out_channels)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        if in_channels == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        feature = torch.relu(x)

        feature = self.conv1(feature)
        feature = self.norm1(feature)

        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)

        connection = feature + self.channel_map(x)
        feature, idx = self.pool(connection)
        return feature, connection, idx


class DeepUNetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=False):
        super(DeepUNetUpSample, self).__init__()
        self.pool = nn.MaxUnpool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = Norm(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = Norm(out_channels)

        self.dropout = nn.Dropout2d(0.5, True) if dropout else None
        if (in_channels // 2) == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d((in_channels // 2),
                                         out_channels,
                                         1,
                                         bias=False)

    def forward(self, x, connection, idx):
        x = self.pool(x, idx)

        feature = torch.relu(x)
        feature = torch.cat([feature, connection], 1)
        feature = self.conv1(feature)
        feature = self.norm1(feature)

        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)

        feature = feature + self.channel_map(x)

        if self.dropout is not None:
            feature = self.dropout(feature)

        return feature
