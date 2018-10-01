"""
Inference model
"""

import torch
from torch.utils.data import Dataset, DataLoader

import os
import glob
import numpy as np
import pandas as pd

from PIL import Image, ImageFilter
import skimage.io as io

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from model import HED
from dataproc import TestDataset

from torchvision import transforms as transforms

from cv_detect import cv2_detect

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load pretrained model
model = HED()
model.to(device)

checkpoint = torch.load(
    'train/HED.pth', map_location=lambda storage, loc: storage)

model.load_state_dict(checkpoint)

filename = './data/tanya.jpg'
origin_image = Image.open(filename)

image = transforms.ToTensor()(origin_image)
image = image.unsqueeze(0)


def gray_scale(img):
    img = img.numpy()[0][0] * 255
    img = img.astype(np.uint8)
    img = np.invert(img)
    return img


s1, s2, s3, s4, s5, s6 = model(image)

s = gray_scale(s1.data)
cv = cv2_detect(filename)
pillow = origin_image.filter(ImageFilter.FIND_EDGES)
pillow = pillow.filter(ImageFilter.SMOOTH_MORE)
pillow = transforms.ToTensor()(pillow)
pillow = pillow.unsqueeze(0)
pillow = gray_scale(pillow.data)

plt.figure(figsize=(12, 12))
plt.imshow(pillow, cmap=cm.Greys_r)
plt.show()

#  fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#
#  axes[0][0].imshow(origin_image)
#  axes[0][1].imshow(s, cmap=cm.Greys_r)
#  axes[1][0].imshow(cv2_detect(filename), cmap=cm.Greys_r)
#  axes[1][1].imshow(np.invert(origin_image.filter(ImageFilter.FIND_EDGES)))
#  plt.show()
