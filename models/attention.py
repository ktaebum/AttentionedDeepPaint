import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 gate_channels,
                 inter_channels=None,
                 bias=True):
        super(AttentionBlock, self).__init__()
        """
        Implementation of Attention block in SegUnet
        """

        if inter_channels is None:
            inter_channels = in_channels // 2

        self.W = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias),
            nn.BatchNorm2d(in_channels),
        )

        # for skip-connection
        self.Wx = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False)

        # for gating
        self.Wg = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        # for internal feature
        self.psi = nn.Conv2d(
            inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        # initialize
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def test(self):
        # model = self(32, 16)
        skip = torch.randn(1, 32, 16, 16)
        g = torch.randn(1, 16, 8, 8)
        result = self(skip, g)
        print(result[0].shape)
        print(result[1].shape)
        pass

    def forward(self, x, g):
        """
        @param g: gated signal (queried)
        @param x: skip connected feature
        """

        # map info inter_channels
        g = self.Wg(g)
        down_x = self.Wx(x)

        g = F.interpolate(g, size=down_x.shape[2:])

        q = self.psi(torch.relu(g + down_x))
        q = torch.sigmoid(q)

        resampled = F.interpolate(q, size=x.shape[2:])
        result = resampled.expand_as(x) * x
        result = self.W(result)

        return result, resampled
