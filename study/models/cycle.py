import torch
import torch.nn as nn

__BLOCK_CFG__ = {
    '6': [32, 64, 128, 128, 128, 128, 128, 128, 64, 32, 3],
    '9': [
        32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3
    ]
}


def set_requires_grad(model, value):
    if not isinstance(value, bool):
        return

    for param in model.parameters():
        param.requires_grad = value


class CycleDiscriminator(nn.Module):
    # 70 x 70 patch GAN
    def __init__(self, norm='batch'):
        super(CycleDiscriminator, self).__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm2d
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d
        else:
            raise KeyError('Invalid normalization function %s' % norm)

        self.dim = 64

        layers = []

        layers.append(nn.Conv2d(3, self.dim, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, True))

        for i in range(1, 3):
            in_channel = self.dim * (2**(i - 1))
            out_channel = self.dim * (2**i)

            layers.append(nn.Conv2d(in_channel, out_channel, 4, 2, 1))
            layers.append(self.norm(out_channel))
            layers.append(nn.LeakyReLU(0.2, True))

        in_channel = out_channel
        out_channel = out_channel * 2

        layers.append(nn.Conv2d(in_channel, out_channel, 4, 1, 1))
        layers.append(self.norm(out_channel))
        layers.append(nn.LeakyReLU(0.2, True))

        layers.append(nn.Conv2d(out_channel, 1, 4, 1, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

        # initialize conv2d weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        return self.layers(image)


class CycleGenerator(nn.Module):

    class ResidualBlock(nn.Module):

        def __init__(self, in_channel, out_channel):
            super(CycleGenerator.ResidualBlock, self).__init__()

            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
            self.bnorm1 = nn.BatchNorm2d(out_channel)

            self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            self.bnorm2 = nn.BatchNorm2d(out_channel)

        def forward(self, x):
            out = x
            out = self.conv1(out)
            out = torch.relu(self.bnorm1(out))

            out = self.conv2(out)
            out = self.bnorm2(out)

            return out + x

    def __init__(self, cfg='9'):
        super(CycleGenerator, self).__init__()
        try:
            self.cfg = __BLOCK_CFG__[cfg]
        except KeyError:
            print('Invalid configuration key %s' % cfg)
            exit()

        self.c7s1_down = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, self.cfg[0], 7, 1, 0),
            nn.InstanceNorm2d(self.cfg[0]), nn.ReLU(True))
        self.down_samples = self._build_down_sample_layers()

        self.residuals = nn.Sequential(*[
            CycleGenerator.ResidualBlock(self.cfg[i], self.cfg[i + 1])
            for i in range(2,
                           len(self.cfg) - 4)
        ])

        self.up_samples = self._build_up_sample_layers()
        self.c7s1_up = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.cfg[-2], self.cfg[-1], 7, 1, 0), nn.Tanh())

        # initialize conv2d weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, image):
        image = self.c7s1_down(image)
        image = self.down_samples(image)
        image = self.residuals(image)
        image = self.up_samples(image)
        image = self.c7s1_up(image)

        return image

    def _build_up_sample_layers(self):
        layers = nn.ModuleList()
        len_cfg = len(self.cfg)

        for i in range(len_cfg - 4, len_cfg - 2):
            in_channel = self.cfg[i]
            out_channel = self.cfg[i + 1]

            block = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1),
                nn.InstanceNorm2d(out_channel), nn.ReLU(True))

            layers.append(block)

        return nn.Sequential(*layers)

    def _build_down_sample_layers(self):
        layers = nn.ModuleList()

        for i in range(1, 3):
            in_channel = self.cfg[i - 1]
            out_channel = self.cfg[i]

            block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                nn.InstanceNorm2d(out_channel), nn.ReLU(True))

            layers.append(block)

        return nn.Sequential(*layers)
