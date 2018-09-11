"""
Conditional GAN code
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from models.ganmodel import cdiscriminator, cgenerator

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
num_epochs = 30
latent_dim = 128


def main():
    mnist_data = input_data.read_data_sets('./data', one_hot=True)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    Z = tf.placeholder(tf.float32, [None, latent_dim])

    with tf.variable_scope('generator'):
        fake_image = cgenerator(Z, Y)

    with tf.variable_scope('discriminator') as scope:
        logit_real = cdiscriminator(2 * X - 1.0, Y)
        scope.reuse_variables()
        logit_fake = cdiscriminator(fake_image, Y)

    # calculate discriminate loss
    label_real = tf.ones_like(logit_real)
    label_dfake = tf.zeros_like(logit_fake)
    loss_discriminate = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit_real, labels=label_real))
    loss_discriminate += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit_fake, labels=label_dfake))

    # calculate generate loss
    label_gfake = tf.ones_like(logit_fake)
    loss_generate = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit_fake, labels=label_gfake))

    # get each trainable variables
    g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              'discriminator')

    # setting var_list prevent backpropagation in other model
    train_discriminate = tf.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.5).minimize(
            loss_discriminate, var_list=d_var)
    train_generate = tf.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.5).minimize(
            loss_generate, var_list=g_var)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, num_epochs + 1):
            for i in range(mnist_data.train.num_examples // batch_size):
                image, labels = mnist_data.train.next_batch(batch_size)

                latent_var = np.random.uniform(-1, 1,
                                               (batch_size, latent_dim))

                loss_d, _ = sess.run(
                    [loss_discriminate, train_discriminate],
                    feed_dict={
                        X: image,
                        Y: labels,
                        Z: latent_var
                    })

                loss_g, _ = sess.run(
                    [loss_generate, train_generate],
                    feed_dict={
                        Y: labels,
                        Z: latent_var
                    })

                if i % 100 == 0:
                    print('D_loss = %f, G_loss = %f' % (loss_d, loss_g))

            # plot test image

            # generate arbitrary label
            y = np.random.randint(0, 10, (16,))

            # draw laten variables
            latent_var = np.random.uniform(-1, 1, (16, latent_dim))

            samples = sess.run(
                fake_image,
                feed_dict={
                    Z: latent_var,
                    Y: sess.run(tf.one_hot(y, 10))
                })

            fig, ax = plt.subplots(4, 4)
            for i in range(4):
                for j in range(4):
                    ax[i][j].set_axis_off()
                    ax[i][j].imshow(samples[4 * i + j].reshape(28, 28))
                    ax[i][j].set_title(y[4 * i + j])
            plt.savefig('test_sample_%d.png' % epoch)
            plt.close()


if __name__ == "__main__":
    main()
