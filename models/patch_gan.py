"""
PathchGAN implementation on Pytorch

Following paper's implementation, using 70 x 70 patchgan
"""

import torch.nn as nn


class PatchGAN(nn.Module):
    def __init__(self, dim=64, norm='batch', sigmoid=True):
        super(PatchGAN, self).__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            raise ValueError('Invalid Normalization')

        self.dim = dim

        layers = nn.ModuleList()

        # 256 x 256
        layers.append(self._building_block(6, self.dim, False))

        # 128 x 128
        layers.append(self._building_block(self.dim, self.dim * 2))

        # 64 x 64
        layers.append(self._building_block(self.dim * 2, self.dim * 4))

        # 63 x 63
        layers.append(
            self._building_block(self.dim * 4, self.dim * 8, stride=1))

        # 62 x62
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.dim * 8, 1, 4, 1, 1),
                nn.Sigmoid() if sigmoid else nn.Sequential()))

        self.layers = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        image = self.layers(image)

        return image

    def _building_block(self, in_channel, out_channel, norm=True, stride=2):
        layers = []
        layers.append(
            nn.Conv2d(in_channel, out_channel, 4, stride=stride, padding=1))
        if norm:
            layers.append(self.norm(out_channel))
        layers.append(nn.LeakyReLU(0.2, True))

        return nn.Sequential(*layers)
