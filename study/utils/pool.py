"""
Image pooling module
got from official pytorch git repo
"""
import random
import torch


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size

        if self.pool_size > 0:
            self.num_images = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []

        for image in images:
            image = torch.unsqueeze(image.data, 0)

            if self.num_images < self.pool_size:
                # not fulled
                self.num_images += 1
                self.images.append(image)
                return_images.append(image)

            else:
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    idx = random.randint(0, self.pool_size - 1)

                    # sample image and replace
                    sampled = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(sampled)
                else:
                    return_images.append(image)

        return_images = torch.cat(return_images, 0)
        return return_images
