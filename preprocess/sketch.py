"""
Sketchfy Module
"""

import torch
import torch.nn as nn

from preprocess.image import re_scale
from PIL import Image, ImageOps, ImageFilter
from torch.utils.serialization import load_lua
from torchvision import transforms
__valid_smooth__ = {'no', 'basic', 'more'}


def simplify(image):
    cache = load_lua('./checkpoints/model_gan.t7')
    model = cache.model
    if torch.cuda.is_available():
        model.cuda()
    image_mean = cache.mean
    image_std = cache.std

    model.evaluate()

    image = image.convert('L')
    width, height = image.size

    padding_width = 8 - (width % 8) if width % 8 != 0 else 0
    padding_height = 8 - (height % 8) if height % 8 != 0 else 0

    image = (
        (transforms.ToTensor()(image) - image_mean) / image_std).unsqueeze(0)

    if padding_width != 0 or padding_height != 0:
        image = nn.ReflectionPad2d((0, padding_width, 0,
                                    padding_height))(image).data

    image = model.forward(image).squeeze(0)
    image = re_scale(image)
    image = transforms.ToPILImage()(image.detach().cpu())

    del model

    return image


def simplify_paired_image(image):
    """
    Simplify Preprocessed image - sketch pair image
    """

    width, height = image.size
    colored = image.crop((0, 0, width // 2, height))
    sketch = image.crop((width // 2, 0, width, height))
    sketch = simplify(sketch)

    simplified_concat = Image.new('RGB', (width, height))

    simplified_concat.paste(colored, (0, 0))
    simplified_concat.paste(sketch, (width // 2, 0))

    return simplified_concat


def get_sketch(image, smooth='basic'):
    """
    From input image file, get sketched (edge) version of input image

    @param image: input image filename or PIL Image Object
    @param smooth: set smooth level ('no', 'basic', 'more')
    """

    if smooth not in __valid_smooth__:
        raise ValueError('Invalid smoothing factor')

    # set filter
    if smooth == 'basic':
        smooth_filter = ImageFilter.SMOOTH
    elif smooth == 'more':
        smooth_filter = ImageFilter.SMOOTH_MORE
    else:
        # no smoothing
        smooth_filter = None

    if isinstance(image, str):
        # input image is a filename, read that image
        image = Image.open(image)

    image = ImageOps.grayscale(image)
    image = image.filter(ImageFilter.FIND_EDGES)
    if smooth_filter is not None:
        image = image.filter(smooth_filter)
    image = ImageOps.invert(image)

    return image
