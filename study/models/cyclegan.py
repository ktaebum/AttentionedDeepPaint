import tensorflow as tf
"""
refer to original CycleGAN paper
"""

# configuration for 6-block generator or 9-block generator
# each list item denotes channels
__BLOCK_CFG__ = {
    '6': [32, 64, 128, 128, 128, 128, 128, 128, 64, 32, 3],
    '9': [
        32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3
    ]
}


class ResidualGenerator:

    def __init__(self, name, resolution=256):
        if resolution >= 256:
            self.cfg = __BLOCK_CFG__['9']
        else:
            self.cfg = __BLOCK_CFG__['6']

        self.kernel_initializer = tf.random_normal_initializer(0, 0.02)
        self.gamma_initializer = tf.random_normal_initializer(1, 0.02)
        self.name = name

    def __call__(self, image):
        """
        image: input image
        """
        cfg_len = len(self.cfg)

        with tf.variable_scope(self.name):
            image = tf.pad(image, ((0, 0), (3, 3), (3, 3), (0, 0)))
            image = tf.layers.conv2d(
                image,
                self.cfg[0], (7, 7),
                padding='valid',
                kernel_initializer=self.kernel_initializer,
                name='c7s1-%d' % self.cfg[0])
            image = tf.contrib.layers.instance_norm(image)
            image = tf.nn.relu(image)

            for i in range(1, 3):
                out_channel = self.cfg[i]
                image = tf.layers.conv2d(
                    image,
                    out_channel, (3, 3), (2, 2),
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    name='d%d' % out_channel)
                image = tf.contrib.layers.instance_norm(image)
                image = tf.nn.relu(image)

            # residual blocks begin
            for layer_num, i in enumerate(range(3, cfg_len - 3), 1):
                out_channel = self.cfg[i]
                image = self._build_residual_block(
                    image, out_channel, 'R%d_%d' % (out_channel, layer_num))

            for i in range(cfg_len - 3, cfg_len - 1):
                out_channel = self.cfg[i]
                image = tf.layers.conv2d_transpose(
                    image,
                    out_channel, (3, 3), (2, 2),
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    name='u%d' % out_channel)
                image = tf.contrib.layers.instance_norm(image)
                image = tf.nn.relu(image)

            image = tf.layers.conv2d_transpose(
                image,
                self.cfg[-1], (7, 7),
                padding='same',
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.tanh,
                name='c7s1-%d' % self.cfg[-1])

            return image

    def _build_residual_block(self,
                              x,
                              out_channel,
                              name,
                              activation=tf.nn.relu):

        out = x

        out = tf.pad(out, ((0, 0), (1, 1), (1, 1), (0, 0)))
        out = self.__conv2d_keepdim(out, out_channel, '%s_conv1' % name)
        out = self.__batch_norm(out, '%s_bnorm1' % name)
        out = tf.nn.relu(out)

        out = tf.pad(out, ((0, 0), (1, 1), (1, 1), (0, 0)))
        out = self.__conv2d_keepdim(out, out_channel, '%s_conv2' % name)
        out = self.__batch_norm(out, '%s_bnorm2' % name)

        return out + x

    def __batch_norm(self, x, name):
        return tf.layers.batch_normalization(
            x,
            name=name,
            gamma_initializer=self.gamma_initializer,
            training=True)

    def __conv2d_keepdim(self, x, out_channel, name):
        # convolutional layer inside of residual block, which keep dimension
        out = tf.layers.conv2d(
            x,
            out_channel, (3, 3), (1, 1),
            padding='valid',
            kernel_initializer=self.kernel_initializer,
            name=name)

        return out
