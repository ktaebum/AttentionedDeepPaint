import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import random

from models.pix import Pix2PixGenerator
from models.patchgan import PatchGAN

from utils.loader import ImageTranslationDataLoader
from utils.losses import gan_loss
from utils.image import random_jittering

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16.0, 9.0)

batch_size = 1
num_epochs = 200
lambd = 100
learning_rate = 0.0002


def main():
    loader = ImageTranslationDataLoader('./data/facades', batch_size)

    A = tf.placeholder(tf.float32, [None, 256, 256, 3])  # real
    B = tf.placeholder(tf.float32, [None, 256, 256, 3])  # facade
    image_holder_A = tf.placeholder(tf.float32, [None, 256, 256, 3])
    aug_A = random_jittering(image_holder_A)
    prob = tf.placeholder_with_default(0.5, shape=())

    generator = Pix2PixGenerator('pixGen', False)
    discriminator = PatchGAN('pixDis', False)

    with tf.variable_scope('generator'):
        fake_image = generator(A, prob)  # make B from A

    with tf.variable_scope('discriminator') as scope:
        logit_real = discriminator(tf.concat([A, B], 3))
        scope.reuse_variables()
        logit_fake = discriminator(tf.concat([A, fake_image], 3))

    g_ce_loss, d_ce_loss = gan_loss(logit_real, logit_fake)

    # add L1 Loss
    l1_loss = tf.reduce_mean(tf.abs(fake_image - B))
    g_loss = g_ce_loss * lambd * l1_loss

    with tf.name_scope('d_train'):
        d_var = [
            var for var in tf.trainable_variables()
            if var.name.startswith("discriminator")
        ]
        d_optim = tf.train.AdamOptimizer(learning_rate, 0.5)
        d_train = d_optim.minimize(d_ce_loss / 2, var_list=d_var)

    with tf.name_scope('g_train'):
        with tf.control_dependencies([d_train]):
            g_var = [
                var for var in tf.trainable_variables()
                if var.name.startswith("generator")
            ]
            g_optim = tf.train.AdamOptimizer(learning_rate, 0.5)
            g_train = g_optim.minimize(g_loss, var_list=g_var)

    def train(sess, last_iter):
        for i, _ in enumerate(range(loader.train_step), last_iter + 1):
            image_B, image_A = loader.get_next_train_batch()

            # random jittering input image
            image_A = sess.run(
                aug_A, feed_dict={
                    image_holder_A: image_A,
                })

            d_loss, _ = sess.run([d_ce_loss, d_train],
                                 feed_dict={
                                     A: image_A,
                                     B: image_B,
                                     prob: 0.5
                                 })

            g_loss_val, l1_loss_val, _, _ = sess.run(
                [g_ce_loss, l1_loss, g_loss, g_train],
                feed_dict={
                    A: image_A,
                    B: image_B,
                    prob: 0.5
                })

            if i % 100 == 0:
                print('D_loss = %f, G_loss = %f, L1_loss = %f' %
                      (d_loss, g_loss_val, l1_loss_val))

        print('Train epoch %d finished' % epoch)
        return i

    def validation(sess):
        image_B, image_A = loader.get_next_val_batch()

        sample = sess.run(fake_image, feed_dict={A: image_A, prob: 0.5})

        # plot random image
        idx = random.randint(0, len(sample) - 1)

        return sample[idx], image_B[idx], image_A[idx]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last_iter = -1
        for epoch in range(1, num_epochs + 1):
            last_iter = train(sess, last_iter)

            sample, real, label = validation(sess)

            fig, ax = plt.subplots(1, 3)

            ax[0].imshow(label)
            ax[0].set_title('label')
            ax[0].axis('off')

            ax[1].imshow(real)
            ax[1].set_title('real')
            ax[1].axis('off')

            ax[2].imshow(sample)
            ax[2].set_title('generated')
            ax[2].axis('off')

            plt.savefig('test_pix2pix_%d.png' % epoch)
            plt.close()


if __name__ == "__main__":
    main()
