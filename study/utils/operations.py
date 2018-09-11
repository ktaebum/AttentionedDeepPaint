import tensorflow as tf


def conv2d(x,
           out_channels,
           kernel_height=3,
           kernel_width=3,
           name='conv2d',
           **kwargs):
    with tf.variable_scope(name):
        out = tf.layers.conv2d(
            x,
            out_channels,
            kernel_size=[kernel_height, kernel_width],
            **kwargs)

        return out


def convt2d(x,
            out_channels,
            kernel_height=3,
            kernel_width=3,
            name='convt2d',
            **kwargs):
    with tf.variable_scope(name):
        out = tf.layers.conv2d_transpose(
            x,
            out_channels,
            kernel_size=(kernel_height, kernel_width),
            **kwargs)

        return out
