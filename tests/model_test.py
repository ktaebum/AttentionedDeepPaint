import unittest

import torch

from models import PatchGAN, Pix2PixGenerator, ResBlock, VggUnet


class ModelTest(unittest.TestCase):
    def test_pix2pix_generator_512(self):
        generator = Pix2PixGenerator()
        shape = (1, 3, 512, 512)
        inputs = torch.randn(*shape)
        outputs = generator(inputs)

        # must output 512 x 512 image
        self.assertEqual(tuple(outputs.shape), shape)

    def test_resblock(self):
        resblock = ResBlock(64, 'reflect', 'batch')
        shape = (1, 64, 32, 32)
        inputs = torch.randn(*shape)
        outputs = resblock(inputs)

        #  generate same shape
        self.assertEqual(tuple(outputs.shape), shape)
