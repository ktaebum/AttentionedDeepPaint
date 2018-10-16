import unittest

import torch

from models import Pix2PixGenerator, ResBlock, HalfResBlock, ResUnet
from models import StylePaintGenerator


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

    def test_style2paint(self):
        generator = StylePaintGenerator()
        image = torch.randn(1, 3, 512, 512)
        style = torch.randn(1, 4096)

        output, guide1, guide2 = generator(image, style)

        self.assertEqual(tuple(output.shape), (1, 3, 512, 512))
        self.assertEqual(tuple(guide1.shape), (1, 3, 512, 512))
        self.assertEqual(tuple(guide2.shape), (1, 3, 512, 512))

    def test_resunet(self):
        generator = ResUnet()
        image = torch.randn(1, 3, 512, 512)
        style = torch.randn(1, 3, 224, 224)

        output, guide1, guide2 = generator(image, style)

        self.assertEqual(tuple(output.shape), (1, 3, 512, 512))
        self.assertEqual(tuple(guide1.shape), (1, 3, 512, 512))
        self.assertEqual(tuple(guide2.shape), (1, 3, 512, 512))
