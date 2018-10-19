"""
Image Preprocessing/Processing Module
"""

import os

import torch

from torchvision import transforms


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
