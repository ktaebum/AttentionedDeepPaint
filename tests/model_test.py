import unittest

import torch

from models import Pix2PixGenerator, ResBlock, HalfResBlock


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

    def test_halfresblock_downsample(self):
        resblock = HalfResBlock(3, 64)
        shape = (1, 3, 16, 16)
        inputs = torch.randn(*shape)
        outputs = resblock(inputs)

        # generate half downsampled shape
        self.assertEqual(tuple(outputs.shape), (1, 64, 8, 8))

    def test_halfresblock_upsample(self):
        resblock = HalfResBlock(64, 3, mode='up')
        shape = (1, 64, 8, 8)
        inputs = torch.randn(*shape)
        outputs = resblock(inputs)

        # generate half upsampled shape
        self.assertEqual(tuple(outputs.shape), (1, 3, 16, 16))
