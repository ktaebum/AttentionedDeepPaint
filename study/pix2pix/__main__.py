import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import random

from utils.facades import FacadeLoader
from models.ganmodel import pix_discriminator, pix_generator
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 8.0)

batch_size = 64
num_epochs = 200
lambd = 100
learning_rate = 0.0002


def main():
    loader = FacadeLoader('./data/facades', batch_size)

    X = tf.placeholder(tf.float32, [None, 256, 256, 3])  # real
    Y = tf.placeholder(tf.float32, [None, 256, 256, 3])  # facade

    # in this version, it does not use this Z
    Z = tf.placeholder(tf.float32, [None, 256, 256, 1])

    with tf.variable_scope('generator'):
        generator = pix_generator(Y, Z)

    with tf.variable_scope('discriminator') as scope:
        logit_real = pix_discriminator(X, Y)
        scope.reuse_variables()
        logit_fake = pix_discriminator(generator, Y)

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

    # add L1 Loss
    loss_generate += lambd * tf.reduce_mean(tf.abs(generator - X))

    # get each trainable variables
    g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              'discriminator')

    # setting var_list prevent backpropagation in other model
    train_discriminate = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.5).minimize(
            loss_discriminate, var_list=d_var)
    train_generate = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.5).minimize(
            loss_generate, var_list=g_var)

    def train(sess, last_iter):
        for i, _ in enumerate(range(loader.train_step), last_iter + 1):
            real, facade = loader.get_next_train_batch()
            z = np.random.normal(size=[*facade.shape[:-1], 1]).astype(
                np.float32)

            loss_d, _ = sess.run([loss_discriminate, train_discriminate],
                                 feed_dict={
                                     X: real,
                                     Y: facade,
                                     Z: z
                                 })

            loss_g, _ = sess.run([loss_generate, train_generate],
                                 feed_dict={
                                     X: real,
                                     Y: facade,
                                     Z: z
                                 })

            # update G twice
            _ = sess.run(
                train_generate, feed_dict={
                    X: real,
                    Y: facade,
                    Z: z
                })

            if i % 10 == 0:
                print('D_loss = %f, G_loss = %f' % (loss_d, loss_g))

        print('Train epoch %d finished' % epoch)
        return last_iter

    def validation(sess):
        real, facade = loader.get_next_val_batch()
        z = np.random.normal(size=[*facade.shape[:-1], 1]).astype(np.float32)

        sample = sess.run(generator, feed_dict={Y: facade, Z: z})

        # plot random image
        idx = random.randint(0, len(sample) - 1)

        return sample[idx], real[idx]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last_iter = -1
        for epoch in range(1, num_epochs + 1):
            last_iter = train(sess, last_iter)

            sample, real = validation(sess)

            fig, ax = plt.subplots(1, 2)

            ax[0].imshow(sample)
            ax[0].set_title('gan')

            ax[1].imshow(real)
            ax[1].set_title('real')

            plt.savefig('test_pix2pix_%d.png' % epoch)
            plt.close()


if __name__ == "__main__":
    main()
