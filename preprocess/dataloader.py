"""
Preprocess train/validation dataset and generate pair image
"""
import os
import glob
import json

from preprocess import scale
from preprocess import make_colorgram_tensor

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """
    Composed of
    (colorized image, sketch image)
    """

    def __init__(self,
                 root='./data/',
                 mode='train',
                 transform=None,
                 color_histogram=False,
                 size=512):
        """
        @param root: data root
        @param mode: set mode (train, test, val)
        @param transform: Image Processing
        @param need_resize: Return 224 resized version of style image
        @param color_histogram: extract color_histogram
        @param size: image crop (or resize) size
        """
        if mode not in {'train', 'val'}:
            raise ValueError('Invalid Dataset. Pick among (train, val)')

        root = os.path.join(root, mode)

        self.is_train = (mode == 'train')
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root, '*.png'))
        self.color_histogram = color_histogram
        self.size = size
        self.color_cache = {}

        if len(self.image_files) == 0:
            # no png file, use jpg
            self.image_files = glob.glob(os.path.join(root, '*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Niko Dataset Get Item
        @param index: index
        Returns:
            if self.color_histogram
            tuple: (imageA == original, imageB == sketch, colors)
            else:
            tuple: (imageA == original, imageB == sketch)

            if self.resize
            resized image will be appended end of the above tuple
        """
        filename = self.image_files[index]
        file_id = filename.split('/')[-1][:-4]

        if self.color_histogram:
            # build colorgram tensor
            color_info = self.color_cache.get(file_id, None)
            if color_info is None:
                with open(
                        os.path.join('./data/colorgram', '%s.json' % file_id),
                        'r') as json_file:
                    # load color info dictionary from json file
                    color_info = json.loads(json_file.read())
                    self.color_cache[file_id] = color_info
            colors = make_colorgram_tensor(color_info)

        image = Image.open(filename)
        image_width, image_height = image.size
        imageA = image.crop((0, 0, image_width // 2, image_height))
        imageB = image.crop((image_width // 2, 0, image_width, image_height))

        # default transforms, pad if needed and center crop 512
        width_pad = self.size - image_width // 2
        if width_pad < 0:
            # do not pad
            width_pad = 0

        height_pad = self.size - image_height
        if height_pad < 0:
            height_pad = 0

        # padding as white
        padding = transforms.Pad((width_pad // 2, height_pad // 2 + 1,
                                  width_pad // 2 + 1, height_pad // 2),
                                 (255, 255, 255))

        # use center crop
        crop = transforms.CenterCrop(self.size)

        imageA = padding(imageA)
        imageA = crop(imageA)

        imageB = padding(imageB)
        imageB = crop(imageB)

        if self.transform is not None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        # scale image into range [-1, 1]
        imageA = scale(imageA)
        imageB = scale(imageB)
        if not self.color_histogram:
            return imageA, imageB
        else:
            return imageA, imageB, colors
