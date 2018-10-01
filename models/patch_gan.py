"""
PathchGAN implementation on Pytorch

Following paper's implementation, using 70 x 70 patchgan
In our case, since we use 512 x 512 image, the final layer of patchgan should
provides 1 x 8 x 8 logit.

Architecture configuration

6 x 512 x 512
64 x 256 x 256
128 x 128 x 128
256 x 64 x 64
512 x 32 x 32
1024 x 16 x 16
1 x 8 x 8
"""

import torch.nn as nn


class PatchGAN(nn.Module):
    def __init__(self, dim=64, norm='batch'):
        super(PatchGAN, self).__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            raise ValueError('Invalid Normalization')

        self.dim = dim

        self.layers = nn.ModuleList()

        self.block1 = self._building_block(6, self.dim)
        self.block2 = self._building_block(self.dim, self.dim * 2)
        self.block3 = self._building_block(self.dim * 2, self.dim * 4)
        self.block4 = self._building_block(self.dim * 4, self.dim * 8)
        self.block5 = self._building_block(self.dim * 8, self.dim * 16)
        self.block6 = nn.Sequential(
            nn.Conv2d(self.dim * 16, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, image):
        image = self.block1(image)
        image = self.block2(image)
        image = self.block3(image)
        image = self.block4(image)
        image = self.block5(image)
        image = self.block6(image)

        return image

    def _building_block(self, in_channel, out_channel):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, 4, 2, 1))
        layers.append(self.norm(out_channel))
        layers.append(nn.LeakyReLU(0.2, True))

        return nn.Sequential(*layers)
