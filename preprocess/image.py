"""
Image Preprocessing/Processing Module
"""

import os

from PIL import ImageOps, ImageFilter, Image

__valid_smooth__ = {'no', 'basic', 'more'}


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


def save_image(image, filename, path='.'):
    """
    save PIL image object as png image file
    @param image: target image
    @param filename: target filename
    @param path: save directory
    """
    extension = '.png'
    if extension != filename[-4:]:
        filename += extension

    path = os.path.join(path, filename)
    image.save(path, "PNG")
