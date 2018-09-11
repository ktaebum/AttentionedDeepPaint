import os
import glob

import random
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from imageio import imread


class FacadeLoader:
    """
    facade image data loader
    """

    def __init__(self, root, batch_size=64):
        train_root = os.path.join(root, "train")
        test_root = os.path.join(root, "test")
        val_root = os.path.join(root, "val")

        self.train_files = glob.glob(os.path.join(train_root, "*.jpg"))
        self.test_files = glob.glob(os.path.join(test_root, "*.jpg"))
        self.val_files = glob.glob(os.path.join(val_root, "*.jpg"))

        random.shuffle(self.train_files)

        self.train_idx = self.test_idx = self.val_idx = 0

        self.len_train = len(self.train_files)
        self.len_test = len(self.test_files)
        self.len_val = len(self.val_files)

        self.batch_size = batch_size

        self.train_step = int(
            np.ceil(len(self.train_files) / self.batch_size))
        self.val_step = int(np.ceil(len(self.val_files) / self.batch_size))

    def get_next_train_batch(self):
        if self.len_train - self.train_idx < self.batch_size:
            # cannot read exact batch size
            files = self.train_files[self.train_idx:]
            self.train_idx = 0
        else:
            files = self.train_files[self.train_idx:self.train_idx +
                                     self.batch_size]
            self.train_idx += self.batch_size

        images = np.array([imread(file) for file in files], dtype=np.float32)

        real = images[:, :, :256]
        facade = images[:, :, 256:]

        real = real / 127.5 - 1.
        facade = facade / 127.5 - 1.

        return real, facade

    def get_next_val_batch(self):
        if self.len_val - self.val_idx < self.batch_size:
            # cannot read exact batch size
            files = self.val_files[self.val_idx:]
            self.val_idx = 0
        else:
            files = self.val_files[self.val_idx:self.val_idx +
                                   self.batch_size]
            self.val_idx += self.batch_size

        images = np.array([imread(file) for file in files], dtype=np.float32)

        real = images[:, :, :256]
        facade = images[:, :, 256:]

        real = real / 127.5 - 1.
        facade = facade / 127.5 - 1.

        return real, facade
