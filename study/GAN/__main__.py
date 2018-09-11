import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from models import discriminator, generator

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128
num_epochs = 10
latent_dim = 128


def main():
    mnist_data = input_data.read_data_sets('./data', one_hot=True)

    with tf.variable_scope('generator'):
        Z = tf.random_uniform(
            shape=[batch_size, latent_dim], minval=-1, maxval=1)
        gen = generator(Z)

    with tf.variable_scope('discriminator') as scope:
        X = tf.placeholder(tf.float32, [None, 28 * 28])
        logit_real = discriminator(2 * X - 1.0)
        scope.reuse_variables()
        logit_fake = discriminator(gen)

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
                image, _ = mnist_data.train.next_batch(batch_size)

                samples = sess.run(gen)

                loss_d, _ = sess.run(
                    [loss_discriminate, train_discriminate],
                    feed_dict={X: image})
                loss_g, _ = sess.run([loss_generate, train_generate])

                if i % 100 == 0:
                    print('D_loss = %f, G_loss = %f' % (loss_d, loss_g))

            # plot test image

            samples = samples[:16]
            fig, ax = plt.subplots(4, 4)
            for i in range(4):
                for j in range(4):
                    ax[i][j].set_axis_off()
                    ax[i][j].imshow(samples[4 * i + j].reshape(28, 28))
            plt.savefig('test_sample_%d.png' % epoch)
            plt.close()


if __name__ == "__main__":
    main()
