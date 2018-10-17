import torch
import random


class ImagePooling:
    """
    Pooling Image based on CycleGAN paper.
    Authors claim that it helps stable training.
    """

    def __init__(self, size=50):
        self.size = size

        if size > 0:
            self.images = []

    def __call__(self, images):
        """
        put images into images list, and get random image
        """
        if self.size == 0:
            # cannot store
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.size:
                # not full yet
                self.images.append(image)
                return_images.append(image)
            else:
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    # pool random image
                    idx = random.randint(0, len(self.images) - 1)

                    # replace
                    return_images.append(self.images[idx])
                    self.images[idx] = image
                else:
                    return_images.append(image)

        return_images = torch.cat(return_images, 0)
        return return_images
