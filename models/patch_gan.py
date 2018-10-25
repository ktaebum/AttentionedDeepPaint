"""
PathchGAN implementation on Pytorch

Following paper's implementation, using 70 x 70 patchgan
In our case, since we use 512 x 512 image, the final layer of patchgan should
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

        self.layers = nn.ModuleList()

        # self.dim x 256 x 256
        self.block1 = self._building_block(6, self.dim, False)

        # self.dim * 2 x 128 x 128
        self.block2 = self._building_block(self.dim, self.dim * 2)

        # self.dim * 4 x 64 x 64
        self.block3 = self._building_block(self.dim * 2, self.dim * 4)

        # self.dim * 8 x 32 x 32
        self.block4 = self._building_block(self.dim * 4, self.dim * 8)

        self.block5 = nn.Sequential(
            nn.Conv2d(self.dim * 8, 1, 4, 1, 1),
            nn.Sigmoid() if sigmoid else nn.Sequential())

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        image = self.block1(image)
        image = self.block2(image)
        image = self.block3(image)
        image = self.block4(image)
        image = self.block5(image)

        return image

    def _building_block(self, in_channel, out_channel, norm=True, stride=2):
        layers = []
        layers.append(
            nn.Conv2d(in_channel, out_channel, 4, stride=stride, padding=1))
        if norm:
            layers.append(self.norm(out_channel))
        layers.append(nn.LeakyReLU(0.2, True))

        return nn.Sequential(*layers)
