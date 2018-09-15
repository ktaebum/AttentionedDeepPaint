import tensorflow as tf

__INITIALIZER__ = tf.random_normal_initializer(0, 0.02)
__GAMMA__ = tf.random_normal_initializer(1, 0.02)


class Pix2PixGenerator:

    def __init__(self, name, reuse):
        self.name = name
        self.reuse = reuse
        self.skip_connections = []

        self.dim = 64

    def _batch_norm(self, image, name):
        return tf.layers.batch_normalization(
            image,
            name=name,
            axis=3,
            training=True,
            epsilon=1e-5,
            momentum=0.1,
            gamma_initializer=__GAMMA__)

    def __call__(self, image, prob):
        with tf.variable_scope(self.name):
            down_cfg = [self.dim, self.dim * 2, self.dim * 4]
            down_cfg += [self.dim * 8 for _ in range(5)]

            for i, cfg in enumerate(down_cfg, 1):
                image = self._downsample(
                    image,
                    cfg,
                    name='down_%d' % i,
                    norm=False if i == 1 else True,
                    activation=tf.nn.leaky_relu)

                self.skip_connections.append(image)

            self.skip_connections = list(reversed(self.skip_connections))[1:]

            up_cfg = [self.dim * 16 for _ in range(4)]
            up_cfg += [self.dim * 8, self.dim * 4, self.dim * 2]

            skip_iter = iter(self.skip_connections)

            for i, cfg in enumerate(up_cfg, 1):
                image = self._upsample(
                    image,
                    cfg,
                    name='up_%d' % i,
                    norm=True,
                    activation=tf.nn.relu)

                if i <= 3:
                    image = tf.nn.dropout(image, prob)

                connection = skip_iter.__next__()
                image = tf.concat([image, connection], 3)

            # final layer
            image = self._upsample(
                image, 3, name='up_final', norm=False, activation=tf.nn.tanh)

            return image

    def _downsample(self,
                    image,
                    out_channel,
                    name,
                    norm=True,
                    activation=None):
        # building block for downsampler
        down_sampled = tf.layers.conv2d(
            image,
            out_channel,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            name=name,
            kernel_initializer=__INITIALIZER__,
            reuse=self.reuse)

        if norm:
            down_sampled = self._batch_norm(down_sampled, '%s_bnorm' % name)

        if activation is None:
            return down_sampled
        else:
            return activation(down_sampled)

    def _upsample(self, image, out_channel, name, norm=True,
                  activation=None):
        up_sampled = tf.layers.conv2d_transpose(
            image,
            out_channel,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            name=name,
            kernel_initializer=__INITIALIZER__,
            reuse=self.reuse)

        if norm:
            up_sampled = self._batch_norm(up_sampled, '%s_bnorm' % name)

        if activation is None:
            return up_sampled
        else:
            return activation(up_sampled)
