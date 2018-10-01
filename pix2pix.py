"""
Pix2Pix Approach
Pix2Pix Model Trainer
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from models import Pix2PixGenerator, PatchGAN

from utils import GANLoss
from utils import get_default_argparser
from utils import load_checkpoints, save_checkpoints

from preprocess import NikoPairedDataset, save_image

from torchvision import transforms

from PIL import Image


def main(args):
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # assign data loader
    train_loader = DataLoader(
        NikoPairedDataset(transform=train_transform),
        shuffle=True,
        batch_size=args.batch_size)
    val_loader = NikoPairedDataset(transform=val_transform, mode='val')

    # assign model
    generator = Pix2PixGenerator(norm='batch', dim=64).to(device)
    discriminator = PatchGAN(norm='batch', dim=64).to(device)

    # assign loss
    gan_loss = GANLoss(True)  # use MSE Loss
    l1_loss = nn.L1Loss()

    # assign optimizer
    optimG = optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))
    optimD = optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))

    if device.type == 'cuda':
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    def train(last_iter):
        for i, datas in enumerate(train_loader, last_iter + 1):
            imageA, imageB = datas
            if args.mode == 'B2A':
                # swap
                imageA, imageB = imageB, imageA
            imageA = imageA.to(device)
            imageB = imageB.to(device)
            fakeB = generator(imageA)

            # proceed Discriminator
            optimD.zero_grad()
            logit_real = discriminator(torch.cat([imageA, imageB], 1))
            d_loss_real = gan_loss(logit_real, True)

            logit_fake = discriminator(torch.cat([imageA, fakeB.detach()], 1))
            d_loss_fake = gan_loss(logit_fake, False)
            d_loss = (d_loss_fake + d_loss_real) * 0.5
            d_loss.backward()
            optimD.step()

            # proceed Generator
            optimG.zero_grad()
            logit_fake = discriminator(torch.cat([imageA, fakeB], 1))
            g_loss_gan = gan_loss(logit_fake, True)
            g_loss_l1 = l1_loss(fakeB, imageB)
            g_loss = g_loss_gan + args.lambd * g_loss_l1
            g_loss.backward()
            optimG.step()

            if args.verbose and i % args.print_every == 0:
                print('Iter %d: d_loss = %f, g_loss = %f, l1_loss = %f' %
                      (i, d_loss, g_loss_gan, g_loss_l1))

        return i

    def validate(epoch=0):
        length = len(val_loader)

        # sample 3 images
        idxs = random.sample(range(0, length - 1), 3)

        sample = Image.new('RGB', (3 * 512, 3 * 512))
        recover = transforms.ToPILImage()

        for i, idx in enumerate(idxs):
            concat = Image.new('RGB', (3 * 512, 512))
            imageA, imageB = val_loader[idx]

            if args.mode == 'B2A':
                imageA, imageB = imageB, imageA

            imageA = imageA.to(device)
            imageB = imageB.to(device)
            fakeB = generator(imageA.unsqueeze(0)).squeeze()

            imageA = ((imageA + 1) * 0.5).detach().cpu()
            imageB = ((imageB + 1) * 0.5).detach().cpu()
            fakeB = ((fakeB + 1) * 0.5).detach().cpu()

            imageA = recover(imageA)
            imageB = recover(imageB)
            fakeB = recover(fakeB)

            concat.paste(imageA, (0, 0))
            concat.paste(imageB, (512, 0))
            concat.paste(fakeB, (2 * 512, 0))

            sample.paste(concat, (0, 0 + 512 * i))

        save_image(sample, 'pix2pix_val_%03d' % epoch,
                   './data/pair_niko/result')

    if args.train:
        last_iter = -1

        for epoch in range(args.num_epochs):
            last_iter = train(last_iter)

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoints(
                    generator, 'pix2pixG', epoch, optimizer=optimG)
                save_checkpoints(
                    discriminator, 'pix2pixD', epoch, optimizer=optimD)
            validate(epoch)
            print('Epoch %d finished' % epoch)

    else:
        validate()


if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument(
        '--lambd',
        help='set l1 loss weight',
        metavar='',
        type=float,
        default=100.)
    parser.add_argument(
        '--mode', help='set mapping mode', metavar='', type=str, default='A2B')
    main(parser.parse_args())
