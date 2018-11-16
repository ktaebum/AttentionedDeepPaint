"""
Image Preprocessing/Processing Module
"""

import os

import torch
import numpy as np

from torchvision import transforms
from PIL import Image
from colorgram import colorgram


def make_colorgram_tensor(color_info, width=512, height=512):
    """
    get color_info dictionary
    Then build color tensor
    """

    topk = len(color_info.keys())

    tensor = torch.ones([topk * 3, width, height])

    for i in range(1, topk + 1):
        # get value
        r, g, b = color_info[str(i)]

        # assign index
        red = (i - 1) * 3
        green = (i - 1) * 3 + 1
        blue = (i - 1) * 3 + 2

        # assign values
        tensor[red] *= r
        tensor[green] *= g
        tensor[blue] *= b

    return scale(tensor / 255.)


def extract_color_histogram(image, topk=4):
    """
    get image
    extract top-k colors except background color
    return (1, 3 * k, image.shape) tensor
    """
    width, height = image.size
    colors = colorgram.extract(image, topk + 1)
    tensor = torch.ones([topk * 3, width, height])
    for i, color in enumerate(colors[1:]):
        red = i * 3
        green = i * 3 + 1
        blue = i * 3 + 2
        tensor[red] *= color.rgb.r
        tensor[green] *= color.rgb.g
        tensor[blue] *= color.rgb.b

    return scale(tensor / 255.)


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


def centor_crop_tensor(image, size=224):
    """
    Center crop image whose type is torch.Tensor (not PIL.Image)

    @param size: target image size (must be small than original size)
    """

    _, _, h, w = image.shape

    h_low = h // 2 - size // 2
    h_high = h // 2 + size // 2
    if h_low < 0 or h_high > h:
        raise IndexError

    w_low = w // 2 - size // 2
    w_high = w // 2 + size // 2
    if w_low < 0 or w_high > w:
        raise IndexError

    image = image[:, :, h_low:h_high, w_low:w_high]
    return image


def scale(image):
    """
    scale image value into [-1, 1]
    """

    return (image * 2) - 1


def re_scale(image):
    """
    re scale scaled image
    """

    return (image + 1) * 0.5


def grayscale_tensor(images, device):
    def grayscale_tensor_(image):
        """
        Grayscale image of tensor
        """

        image = image.detach().cpu()
        image = re_scale(image)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        image = transform(image)
        return scale(image)

    return torch.stack(list(map(grayscale_tensor_, images))).to(device)


def black2white(image, threshold=30):
    """
    Given PIL image, find black-padded lines and convert it into white padding
    (For easy training)
    """
    image = image.convert('RGB')
    image = np.array(image)
    width, height, _ = image.shape
    for w in range(width):
        channel = np.multiply.reduce((image[w] <= threshold).reshape(-1))
        is_black = (channel == 1)
        if is_black:
            image[w] = 255.
    for h in range(height):
        channel = np.multiply.reduce((image[:, h] <= threshold).reshape(-1))
        is_black = (channel == 1)
        if is_black:
            image[:, h] = 255.
    return Image.fromarray(image)
