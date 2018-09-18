import os
import glob

import random
import subprocess
import numpy as np

import torch
import torchvision.transforms as transforms

from PIL import Image


class ImageTranslationDataLoader:
    """
    image translation dataset loader
    """

    def __init__(self, root, batch_size=1):
        data_name = root.split('/')[-1]
        if not os.path.exists(root):
            print('No %s data exists!' % data_name)
            print('Download...')
            subprocess.call(['sh', 'download_data.sh', data_name])
        train_root = os.path.join(root, "train")
        val_root = os.path.join(root, "val")

        self.train_files = glob.glob(os.path.join(train_root, "*.jpg"))
        self.val_files = glob.glob(os.path.join(val_root, "*.jpg"))

        random.shuffle(self.train_files)

        self.train_idx = self.val_idx = 0

        self.len_train = len(self.train_files)
        self.len_val = len(self.val_files)

        self.batch_size = batch_size

        self.train_step = int(
            np.ceil(len(self.train_files) / self.batch_size))
        self.val_step = int(np.ceil(len(self.val_files) / self.batch_size))

        self.transforms_ = transforms.Compose([
            transforms.Resize(286, Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_next_train_batch(self):
        if self.len_train - self.train_idx < self.batch_size:
            # cannot read exact batch size
            files = self.train_files[self.train_idx:]
            self.train_idx = 0
        else:
            files = self.train_files[self.train_idx:self.train_idx +
                                     self.batch_size]
            self.train_idx += self.batch_size
            if self.train_idx == self.len_train:
                self.train_idx = 0

        image_A, image_B = [], []
        for file in files:
            AB = Image.open(file)
            A = AB.crop((0, 0, 256, 256))
            B = AB.crop((256, 0, 512, 256))
            image_A.append(self.transforms_(A))
            image_B.append(self.transforms_(B))

        image_A = torch.stack(image_A)
        image_B = torch.stack(image_B)

        return image_A, image_B

    def get_next_val_batch(self):
        if self.len_val - self.val_idx < self.batch_size:
            # cannot read exact batch size
            files = self.val_files[self.val_idx:]
            self.val_idx = 0
        else:
            files = self.val_files[self.val_idx:self.val_idx +
                                   self.batch_size]
            self.val_idx += self.batch_size
            if self.val_idx == self.len_val:
                self.val_idx = 0

        image_A, image_B = [], []
        for file in files:
            AB = Image.open(file)
            A = AB.crop((0, 0, 256, 256))
            B = AB.crop((256, 0, 512, 256))
            image_A.append(self.transforms_(A))
            image_B.append(self.transforms_(B))

        image_A = torch.stack(image_A)
        image_B = torch.stack(image_B)

        return image_A, image_B
