"""
Improvement of DeepPaint

No guide, add attention
"""

import torch
import torch.nn as nn

Norm = nn.InstanceNorm2d
Norm = nn.BatchNorm2d


class AttentionPaintGenerator(nn.Module):
    """
    Use Unet & SegNet feature
    """

    def __init__(self, bias=True):
        super(AttentionPaintGenerator, self).__init__()

        self.bias = bias
        self.dim = 64
        self.bridge_channel = 4096
        self.relu = nn.ReLU(True)

        self.down_sampler = self._down_sample()
        self.up_sampler = self._up_sample()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def test(self):
        x = torch.randn(1, 3, 512, 512)
        c = torch.randn(1, 12, 512, 512)
        s = torch.randn(1, 4096)
        print(self.forward(x, c, s).shape)

    def forward(self, image, colors, style):
        """
        style shoud be (batch_size x 4096)
        """

        # unsqueeze style
        style = style.unsqueeze(-1).unsqueeze(-1)

        cache = []
        image = torch.cat([image, colors], 1)

        for i, layer in enumerate(self.down_sampler):
            image, connection, idx = layer(image)
            cache.append((connection, idx))

        cache = list(reversed(cache))
        image = image + style

        for i, (layer, (connection, idx)) in enumerate(
                zip(self.up_sampler, cache)):
            image = layer(image, connection, idx)

        return image

    def _down_sample(self):
        layers = nn.ModuleList()

        # 256
        layers.append(AttentionPaintDownSample(15, self.dim, self.bias))

        # 128
        layers.append(
            AttentionPaintDownSample(self.dim, self.dim * 2, self.bias))

        # 64
        layers.append(
            AttentionPaintDownSample(self.dim * 2, self.dim * 4, self.bias))

        # 32
        layers.append(
            AttentionPaintDownSample(self.dim * 4, self.dim * 8, self.bias))

        # 16
        layers.append(
            AttentionPaintDownSample(self.dim * 8, self.dim * 8, self.bias))

        # 8
        layers.append(
            AttentionPaintDownSample(self.dim * 8, self.bridge_channel,
                                     self.bias))

        return layers

    def _up_sample(self):
        layers = nn.ModuleList()
        layers.append(
            AttentionPaintUpSample(self.bridge_channel * 2, self.dim * 8,
                                   self.bias, True))
        layers.append(
            AttentionPaintUpSample(self.dim * 8 * 2, self.dim * 8, self.bias,
                                   True))
        layers.append(
            AttentionPaintUpSample(self.dim * 8 * 2, self.dim * 4, self.bias,
                                   True))
        layers.append(
            AttentionPaintUpSample(self.dim * 4 * 2, self.dim * 2, self.bias))
        layers.append(
            AttentionPaintUpSample(self.dim * 2 * 2, self.dim * 1, self.bias))
        layers.append(
            AttentionPaintUpSample(
                self.dim * 1 * 2, 3, self.bias, is_last=True))
        return layers


class AttentionPaintDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(AttentionPaintDownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm = Norm(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        connection = self.norm(x)
        x = self.activation(connection)
        x, idx = self.pool(x)
        return x, connection, idx


class AttentionPaintUpSample(nn.Module):
    """
    It's duplicate of DeepPaintUpSample now,
    but attention features will be added
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 dropout=False,
                 is_last=False):
        super(AttentionPaintUpSample, self).__init__()
        self.pool = nn.MaxUnpool2d(2, 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm = Norm(out_channels)
        self.dropout = nn.Dropout2d(0.5, True) if dropout else None
        self.activation = nn.ReLU(True)
        self.is_last = is_last

    def forward(self, x, connection, idx):
        x = self.pool(x, idx)
        x = torch.cat([x, connection], 1)
        x = self.conv(x)
        if self.is_last:
            return torch.tanh(x)
        else:
            x = self.norm(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.activation(x)
            return x
