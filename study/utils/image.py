"""
got code from 
https://github.com/yenchenlin/pix2pix-tensorflow/blob/master/utils.py
"""
import tensorflow as tf
import random

from imageio import imread
import numpy as np
import scipy


def preprocess(image, scale=127.5, bias=1):
    # scale image value
    return image / scale - bias


def random_jittering(image_holder):
    resized = tf.image.resize_image_with_pad(image_holder, 286, 286)
    cropped = tf.image.resize_image_with_crop_or_pad(resized, 256, 256)

    if random.random() > 0.5:
        cropped = tf.image.random_flip_left_right(cropped)

    return cropped


def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(
        img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A / 127.5 - 1.
    img_B = img_B / 127.5 - 1.

    return img_A, img_B


def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w / 2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B


def preprocess_A_and_B(img_A,
                       img_B,
                       load_size=286,
                       fine_size=256,
                       flip=True,
                       is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
        img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B


def inverse_transform(images):
    return (images + 1.) / 2.
