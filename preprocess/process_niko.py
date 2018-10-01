"""
Preprocess niko dataset and generate pair image
"""
import os
import glob

from . import get_sketch, save_image

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class NikoPairedDataset(Dataset):
    """
    Niko Manga Paired Dataset
    Composed of
    (colorized image, sketch image)
    """

    def __init__(self, root='./data/pair_niko', mode='train', transform=None):
        if mode not in {'train', 'val', 'test'}:
            raise ValueError('Invalid Dataset. Pick among (train, val, test)')

        root = os.path.join(root, mode)

        self.is_train = (mode == 'train')
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root, '*.png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Niko Dataset Get Item
        @param index: index
        Returns:
            tuple: (imageA == original, imageB == sketch)
        """

        image = Image.open(self.image_files[index])

        image_width, image_height = image.size
        imageA = image.crop((0, 0, image_width // 2, image_height))
        imageB = image.crop((image_width // 2, 0, image_width, image_height))

        # default transforms, pad if needed and center crop 512
        width_pad = 512 - image_width // 2
        height_pad = 512 - image_height

        padding = transforms.Pad((0, height_pad, width_pad, 0),
                                 (255, 255, 255))

        imageA = padding(imageA)
        imageA = imageA.crop((0, 0, 512, 512))

        imageB = padding(imageB)
        imageB = imageB.crop((0, 0, 512, 512))

        if self.transform is not None:
            image = self.transform(image)

        return imageA, imageB


if __name__ == "__main__":
    # here comes generating sketch images
    img_files = glob.glob('./data/niko/**/*.jpg', recursive=True)
    for file in img_files:
        filename = file.split('/')[-1][:-4]
        print('Processing %s' % file)
        original = Image.open(file)
        sketch = get_sketch(file, 'more')

        width, height = sketch.size

        concat = Image.new('RGB', (2 * width, height))

        concat.paste(original, (0, 0))
        concat.paste(sketch, (width, 0))

        save_image(concat, filename, './data/pair_niko')
