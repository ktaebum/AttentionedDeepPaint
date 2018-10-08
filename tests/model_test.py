import unittest

import torch

from models import PatchGAN, Pix2PixGenerator, ResBlock, VggUnet


class TestVGGUNet(unittest.TestCase):
    def test_pix2pix_generator_512(self):
        generator = Pix2PixGenerator()
        shape = (1, 3, 512, 512)
        inputs = torch.randn(*shape)
        outputs = generator(inputs)

        self.assertEqual(tuple(outputs.shape), shape)
