import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import random
import numpy as np

from utils.args import get_default_argparser
from utils.loader import ImageTranslationDataLoader
from utils.pool import ImagePool
from utils.plot import plot_image

from models.cycle import CycleGenerator, CycleDiscriminator


class LambdaLR():

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = ImageTranslationDataLoader('./data/facades', args.batch_size)

    generator_A = CycleGenerator().to(device)  # generator which makes fakeA
    generator_B = CycleGenerator().to(device)  # generator which makes fakeB

    discriminator_A = CycleDiscriminator().to(
        device)  # discriminator for real A and fake A
    discriminator_B = CycleDiscriminator().to(
        device)  # discriminator for real B and fake B

    # optimizer for generator
    optim_G = optim.Adam(
        list(generator_A.parameters()) + list(generator_B.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))

    # optimizer for discriminator
    optim_D_A = optim.Adam(
        discriminator_A.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))
    optim_D_B = optim.Adam(
        discriminator_B.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))

    # learning_rate schedular
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optim_G,
        lr_lambda=LambdaLR(args.num_epochs, 0, args.num_epochs // 2).step)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optim_D_A,
        lr_lambda=LambdaLR(args.num_epochs, 0, args.num_epochs // 2).step)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optim_D_B,
        lr_lambda=LambdaLR(args.num_epochs, 0, args.num_epochs // 2).step)

    cycle_loss_function = nn.L1Loss().to(device)
    gan_loss_function = nn.MSELoss().to(device)

    if device.type == 'cuda':
        cudnn.benchmark = True
        generator_A = nn.DataParallel(generator_A)
        generator_B = nn.DataParallel(generator_B)
        discriminator_A = nn.DataParallel(discriminator_A)
        discriminator_B = nn.DataParallel(discriminator_B)

    if args.train:
        fake_A_pool = ImagePool(args.pool_size)
        fake_B_pool = ImagePool(args.pool_size)

    def sample(epoch):
        """
        Sample image from validation set
        """
        image_A, image_B = loader.get_next_val_batch()

        image_A = torch.FloatTensor(image_A).to(device)
        image_B = torch.FloatTensor(image_B).to(device)

        fake_A = generator_A(image_B)
        fake_B = generator_B(image_A)

        idx = random.randint(0, len(image_A) - 1)

        image_A = image_A[idx]
        image_B = image_B[idx]
        fake_A = fake_A[idx]
        fake_B = fake_B[idx]

        A = torch.cat([image_A, fake_A], 2)
        B = torch.cat([image_B, fake_B], 2)

        AB = torch.cat([A, B], 1).detach().cpu().numpy()
        AB = np.transpose(AB, (1, 2, 0))

        plot_image(AB, 'test_cycle_%d.png' % epoch)

    def train():
        for i, _ in enumerate(range(loader.train_step), 1):
            image_A, image_B = loader.get_next_train_batch()

            image_A = torch.FloatTensor(image_A).to(device)
            image_B = torch.FloatTensor(image_B).to(device)

            # update generator
            optim_G.zero_grad()

            fake_A = generator_A(image_B)
            fake_B = generator_B(image_A)

            cycled_A = generator_A(fake_B)
            cycled_B = generator_B(fake_A)

            logit_A_fake = discriminator_A(fake_A)
            logit_B_fake = discriminator_B(fake_B)
            loss_G_A = gan_loss_function(logit_A_fake,
                                         torch.ones_like(logit_A_fake))
            loss_G_B = gan_loss_function(logit_B_fake,
                                         torch.ones_like(logit_B_fake))
            loss_cycle_A = cycle_loss_function(cycled_A, image_A)
            loss_cycle_B = cycle_loss_function(cycled_B, image_B)

            loss_G = loss_G_A + loss_G_B + args.lambda_weight * loss_cycle_A + args.lambda_weight * loss_cycle_B
            loss_G.backward()
            optim_G.step()

            # update discriminator A
            optim_D_A.zero_grad()

            fake_A = fake_A_pool.query(fake_A.detach().cpu()).to(device)
            logit_A_real = discriminator_A(image_A)
            logit_A_fake = discriminator_A(fake_A)

            loss_D_A_fake = gan_loss_function(logit_A_fake,
                                              torch.zeros_like(logit_A_fake))
            loss_D_A_real = gan_loss_function(logit_A_real,
                                              torch.ones_like(logit_A_real))
            loss_D_A = (loss_D_A_fake + loss_D_A_real) * 0.5
            loss_D_A.backward()
            optim_D_A.step()

            # update discriminator B
            optim_D_B.zero_grad()
            fake_B = fake_B_pool.query(fake_B.detach().cpu()).to(device)
            logit_B_real = discriminator_B(image_B)
            logit_B_fake = discriminator_B(fake_B)

            loss_D_B_fake = gan_loss_function(logit_B_fake,
                                              torch.zeros_like(logit_B_fake))
            loss_D_B_real = gan_loss_function(logit_B_real,
                                              torch.ones_like(logit_B_real))
            loss_D_B = (loss_D_B_fake + loss_D_B_real) * 0.5
            loss_D_B.backward()
            optim_D_B.step()

            if args.verbose and i % args.print_every == 0:
                print(
                    'Loss G_A = %f, Loss G_B = %f, Loss D_A = %f, Loss = D_B = %f, Loss Cycle = %f'
                    % (loss_G_A.item(), loss_G_B.item(), loss_D_A.item(),
                       loss_D_B.item(),
                       loss_cycle_A.item() + loss_cycle_B.item()))

    if args.train:
        for epoch in range(1, args.num_epochs + 1):
            train()
            sample(epoch)
            if args.verbose:
                print('Epoch %d finished' % epoch)

            scheduler_G.step()
            scheduler_D_A.step()
            scheduler_D_B.step()


if __name__ == "__main__":
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
    parser.add_argument(
        '--pool-size',
        help='set history image pooling size',
        metavar='',
        type=int,
        default=50)

    main(parser.parse_args())
