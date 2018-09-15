"""
Implementation of PATCHGAN
"""

import tensorflow as tf

__INITIALIZER__ = tf.random_normal_initializer(0, 0.02)
__GAMMA__ = tf.random_normal_initializer(1, 0.02)


class PatchGAN:

    def __init__(self, name, reuse=False):
        self.name = name
        self.reuse = reuse

    def _batch_norm(self, image, name):
        return tf.layers.batch_normalization(
            image,
            name=name,
            axis=3,
            training=True,
            epsilon=1e-5,
            momentum=0.1,
            gamma_initializer=__GAMMA__)

    def __call__(self, image):
        """
        Discriminate given [batch_size, 256, 256, 6] image
        """

        with tf.variable_scope(self.name):
            # layer 1
            image = tf.layers.conv2d(
                image,
                64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name='C64',
                activation=tf.nn.leaky_relu,
                kernel_initializer=__INITIALIZER__,
                reuse=self.reuse)

            # layer 2
            image = tf.layers.conv2d(
                image,
                128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name='C128',
                kernel_initializer=__INITIALIZER__,
                reuse=self.reuse)
            image = self._batch_norm(tf.nn.leaky_relu(image), name='bnorm_1')

            # layer 3
            image = tf.layers.conv2d(
                image,
                256,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name='C256',
                kernel_initializer=__INITIALIZER__,
                reuse=self.reuse)
            image = self._batch_norm(tf.nn.leaky_relu(image), name='bnorm_2')

            # layer 4
            image = tf.layers.conv2d(
                image,
                512,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name='C512',
                kernel_initializer=__INITIALIZER__,
                reuse=self.reuse)
            image = self._batch_norm(tf.nn.leaky_relu(image), name='bnorm_3')

            # layer 5
            image = tf.layers.conv2d(
                image,
                1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                name='CFinal',
                activation=tf.nn.sigmoid,
                reuse=self.reuse)

            return image
