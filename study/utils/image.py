import tensorflow as tf
import random


def preprocess(image, scale=127.5, bias=1):
    # scale image value
    return image / scale - bias


def random_jittering(image_holder):
    resized = tf.image.resize_image_with_pad(image_holder, 286, 286)
    cropped = tf.image.resize_image_with_crop_or_pad(resized, 256, 256)

    if random.random() > 0.5:
        cropped = tf.image.random_flip_left_right(cropped)

    return cropped
