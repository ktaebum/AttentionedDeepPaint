import tensorflow as tf


def pix_generator(image, z):
    # now image shape becomes [batch_size, 256, 256, 4]
    # image = tf.concat([image, z], 3)

    out_channels = [32 * (2**i) for i in range(6)]
    for _ in range(2):
        out_channels.append(out_channels[-1])

    shortcuts = []
    for i, out_channel in enumerate(out_channels, 1):
        image = tf.layers.conv2d(
            image,
            out_channel, (3, 3), (2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            name='pix_g_conv%d' % i)

        image = tf.layers.batch_normalization(image)

        if i < len(out_channels):
            shortcuts.append(image)

    shortcuts = list(reversed(shortcuts))
    for i, out_channel in enumerate(reversed(out_channels), 1):
        if i == len(out_channels):
            image = tf.layers.conv2d_transpose(
                image,
                3, (3, 3), (2, 2),
                padding='same',
                activation=tf.nn.relu,
                name='pix_g_convt%d' % i)
        else:
            image = tf.layers.conv2d_transpose(
                image,
                out_channel, (3, 3), (2, 2),
                padding='same',
                activation=tf.nn.relu,
                name='pix_g_convt%d' % i)

        if i < len(out_channels):
            image = tf.layers.dropout(
                tf.layers.batch_normalization(image), 0.5)
            image = tf.concat([image, shortcuts[i - 1]], 3)

    return tf.nn.tanh(image)


def pix_discriminator(real, facade):

    # image shape is [batch_size, 256, 256, 6]
    image = tf.concat([real, facade], 3)
    out_channels = [16 * (2**i) for i in range(4)]

    for i, out_channel in enumerate(reversed(out_channels), 1):
        image = tf.layers.conv2d(
            image,
            out_channel, (3, 3), (2, 2),
            activation=tf.nn.leaky_relu,
            name='pix_d_convt%d' % i)

    image = tf.layers.dense(tf.layers.flatten(image), 1, name='pix_d_linear')

    return tf.nn.sigmoid(image)


def cgenerator(z, y, units=[1024, 1024, 784]):
    """
    Generator model for conditional GAN
    """

    # add y-information to z
    out = tf.concat([z, y], 1)

    for u, unit in enumerate(units, 1):
        activation = tf.nn.tanh if u == len(units) else tf.nn.relu
        out = tf.layers.dense(
            out, unit, activation=activation, name='cgenerator_layer_%d' % u)

    return out


def cdiscriminator(x, y, units=[256, 256, 1]):
    out = tf.concat([x, y], 1)
    for u, unit in enumerate(units, 1):
        activation = None if u == len(units) else tf.nn.leaky_relu
        out = tf.layers.dense(
            out,
            unit,
            activation=activation,
            name='cdiscriminator_layer_%d' % u)

    return out


def generator(z, units=[1024, 1024, 784]):
    """
    get latent variables z
    return generated image of mnist

    using just fully-connected layer
    """

    out = z
    for u, unit in enumerate(units, 1):
        activation = tf.nn.tanh if u == len(units) else tf.nn.relu
        out = tf.layers.dense(
            out, unit, activation=activation, name='generator_layer_%d' % u)

    return out


def discriminator(image, units=[256, 256, 1]):
    """
    get image
    return discriminator's result (True or False)

    using just fully-connected layer
    """

    out = image
    for u, unit in enumerate(units, 1):
        activation = None if u == len(units) else tf.nn.leaky_relu
        out = tf.layers.dense(
            out,
            unit,
            activation=activation,
            name='discriminator_layer_%d' % u)

    return out
