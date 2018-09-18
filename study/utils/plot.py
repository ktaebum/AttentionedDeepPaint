import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
from utils.image import inverse_transform


def plot_image(image, name):
    """
    Get image tensor, plot it
    """
    image = (image + 1.) / 2.0 * 255.0
    image = image.astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(name)
    plt.close()
    pass
