import matplotlib
matplotlib.use('Agg')

import random
import tensorflow as tf
import matplotlib.pyplot as plt

from models.cyclegan import ResidualGenerator
from models.patchgan import PatchGAN

from utils.losses import gan_loss_cycle, cycle_loss
from utils.facades import ImageTranslationDataLoader
from utils.args import get_default_argparser
from utils.image import inverse_transform

plt.rcParams['figure.figsize'] = (12.0, 8.0)

parser = get_default_argparser()
parser.add_argument(
    '--beta1',
    help='set beta1 value of adam optimizer',
    metavar='',
    type=float,
    default=0.5)

parser.add_argument(
    '--lambda-weight',
    help='set lambda (weight of cycle loss)',
    metavar='',
    type=float,
    default=10.)

args = parser.parse_args()

beta1 = args.beta1
batch_size = args.batch_size
lambd = args.lambda_weight
learning_rate = args.learning_rate
num_epochs = args.num_epochs
print_every = args.print_every
verbose = args.verbose
is_train = args.train


def main():

    # loader = ImageTranslationDataLoader('./data/edges2shoes', batch_size)
    loader = ImageTranslationDataLoader('./data/facades', batch_size)

    A = tf.placeholder(tf.float32, [None, 256, 256, 3])
    B = tf.placeholder(tf.float32, [None, 256, 256, 3])

    generator = ResidualGenerator('cycleGen')
    discriminator = PatchGAN('cycleDis')

    with tf.variable_scope('generator'):
        with tf.variable_scope('A2B'):
            fake_B = generator(A)

    with tf.variable_scope('generator'):
        with tf.variable_scope('B2A') as g_scope:
            fake_A = generator(B)
            g_scope.reuse_variables()
            cycled_A = generator(fake_B)

        with tf.variable_scope('A2B') as g_scope:
            g_scope.reuse_variables()
            cycled_B = generator(fake_A)

    with tf.variable_scope('discriminator'):
        with tf.variable_scope('B') as d_scope:
            logit_B_real = discriminator(B)
            d_scope.reuse_variables()
            logit_B_fake = discriminator(fake_B)

    with tf.variable_scope('discriminator'):
        with tf.variable_scope('A') as d_scope:
            logit_A_real = discriminator(A)
            d_scope.reuse_variables()
            logit_A_fake = discriminator(fake_A)

    # calculate losses
    g_loss_A2B, d_loss_A2B = gan_loss_cycle(logit_B_real, logit_B_fake)
    g_loss_B2A, d_loss_B2A = gan_loss_cycle(logit_A_real, logit_A_fake)
    c_loss = cycle_loss(cycled_A, A, cycled_B, B)

    g_loss_obj = g_loss_A2B + g_loss_B2A + lambd * c_loss
    d_loss_obj = (d_loss_A2B + d_loss_B2A) / 2  # slow down d_train

    # get variables
    g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              'discriminator')

    print('len g_var = %d, len d_var = %d' % (len(g_var), len(d_var)))

    # set train objective
    g_train = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1).minimize(
            g_loss_obj, var_list=g_var)
    d_train = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1).minimize(
            d_loss_obj, var_list=d_var)

    def train(sess, last_iter):
        for i, _ in enumerate(range(loader.train_step), last_iter + 1):
            image_A, image_B = loader.get_next_train_batch()

            loss_d_A, loss_d_B, _ = sess.run(
                [d_loss_B2A, d_loss_A2B, d_train],
                feed_dict={
                    A: image_A,
                    B: image_B,
                })

            loss_g_A, loss_g_B, loss_c, _ = sess.run(
                [g_loss_B2A, g_loss_A2B, c_loss, g_train],
                feed_dict={
                    A: image_A,
                    B: image_B,
                })

            if i % print_every == 0:
                print(
                    'D_loss_A = %f, D_loss_B = %f, G_loss_A = %f, G_loss_B = %f, C_loss = %f'
                    % (loss_d_A, loss_d_B, loss_g_A, loss_g_B, loss_c))

        print('Train epoch %d finished' % epoch)
        return i

    def validation(sess):
        image_A, image_B = loader.get_next_val_batch()

        B_sample = sess.run(fake_B, feed_dict={A: image_A})
        A_sample = sess.run(fake_A, feed_dict={B: image_B})

        # plot random image
        idx = random.randint(0, len(B_sample) - 1)

        return image_B[idx], B_sample[idx], image_A[idx], A_sample[idx]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last_iter = -1
        for epoch in range(1, num_epochs + 1):
            last_iter = train(sess, last_iter)

            B_real, B_fake, A_real, A_fake = validation(sess)

            fig, ax = plt.subplots(2, 2)

            ax[0][0].imshow(inverse_transform(B_real))
            ax[0][0].set_title('REAL IMAGE', size=18)
            ax[0][0].set_axis_off()
            ax[0][1].imshow(inverse_transform(B_fake))
            ax[0][1].set_title('GAN IMAGE', size=18)
            ax[0][1].set_axis_off()

            ax[1][0].imshow(inverse_transform(A_real))
            ax[1][0].set_axis_off()
            ax[1][1].imshow(inverse_transform(A_fake))
            ax[1][1].set_axis_off()

            plt.savefig('test_cycle_%d.png' % epoch)
            plt.close()


if __name__ == "__main__":
    main()
