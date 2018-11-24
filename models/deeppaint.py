import torch
import torch.nn as nn

Norm = nn.InstanceNorm2d
Norm = nn.BatchNorm2d


class DeepPaintDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeepPaintDownSample, self).__init__()
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


class DeepPaintUpSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 dropout=False,
                 is_last=False):
        super(DeepPaintUpSample, self).__init__()
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


class DeepPaintGuideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, is_last=False):
        super(DeepPaintGuideBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm = Norm(out_channels)
        self.activation = nn.ReLU(True)
        self.is_last = is_last

    def forward(self, x, idx):
        x = self.unpool(x, idx)
        x = self.conv(x)
        if self.is_last:
            return torch.tanh(x)
        else:
            x = self.norm(x)
            x = self.activation(x)
            return x


class DeepPaintGenerator(nn.Module):
    """
    Use Unet & SegNet feature
    """

    def __init__(self, bias=True):
        super(DeepPaintGenerator, self).__init__()

        self.bias = bias
        self.dim = 64
        self.relu = nn.ReLU(True)

        self.down_sampler = self._down_sample()
        self.up_sampler = self._up_sample()
        self.guide1 = self._guide_decoder()
        self.guide2 = self._guide_decoder()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image, colors, styles):
        cache = []
        image = torch.cat([image, colors], 1)

        for i, layer in enumerate(self.down_sampler):
            if i == len(self.down_sampler) - 1:
                guide1 = self.forward_guide1(image,
                                             list(map(lambda x: x[1], cache)))
            image, connection, idx = layer(image)
            cache.append((connection, idx))

        cache = list(reversed(cache))

        for i, (layer, (connection, idx)) in enumerate(
                zip(self.up_sampler, cache)):
            image = layer(image, connection, idx)
            if i == 0:
                image = image + styles[1]
                guide2 = self.forward_guide2(image,
                                             list(map(lambda x: x[1], cache)))
            elif i == 1:
                image = image + styles[0]

        return image, guide1, guide2

    def forward_guide1(self, x, idxs):
        idxs = list(reversed(idxs))
        for layer, idx in zip(self.guide1, idxs):
            x = layer(x, idx)
        return x

    def forward_guide2(self, x, idxs):
        idxs = idxs[1:]
        for layer, idx in zip(self.guide2, idxs):
            x = layer(x, idx)
        return x

    def _guide_decoder(self):
        layers = nn.ModuleList()
        layers.append(
            DeepPaintGuideBlock(self.dim * 8, self.dim * 8, self.bias))
        layers.append(
            DeepPaintGuideBlock(self.dim * 8, self.dim * 4, self.bias))
        layers.append(
            DeepPaintGuideBlock(self.dim * 4, self.dim * 2, self.bias))
        layers.append(
            DeepPaintGuideBlock(self.dim * 2, self.dim * 1, self.bias))
        layers.append(DeepPaintGuideBlock(self.dim * 1, 3, self.bias, True))
        return layers

    def _down_sample(self):
        layers = nn.ModuleList()

        # 256
        layers.append(DeepPaintDownSample(15, self.dim, self.bias))

        # 128
        layers.append(DeepPaintDownSample(self.dim, self.dim * 2, self.bias))

        # 64
        layers.append(
            DeepPaintDownSample(self.dim * 2, self.dim * 4, self.bias))

        # 32
        layers.append(
            DeepPaintDownSample(self.dim * 4, self.dim * 8, self.bias))

        # 16
        layers.append(
            DeepPaintDownSample(self.dim * 8, self.dim * 8, self.bias))

        # 8
        layers.append(
            DeepPaintDownSample(self.dim * 8, self.dim * 8, self.bias))

        return layers

    def _up_sample(self):
        layers = nn.ModuleList()
        layers.append(
            DeepPaintUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepPaintUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(
            DeepPaintUpSample(self.dim * 8 * 2, self.dim * 4, self.bias, True))
        layers.append(
            DeepPaintUpSample(self.dim * 4 * 2, self.dim * 2, self.bias))
        layers.append(
            DeepPaintUpSample(self.dim * 2 * 2, self.dim * 1, self.bias))
        layers.append(
            DeepPaintUpSample(self.dim * 1 * 2, 3, self.bias, is_last=True))
        return layers
