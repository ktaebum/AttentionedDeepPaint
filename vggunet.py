"""
Naive Vgg + Unet Approach
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from models import VggUnet, PatchGAN

from utils import GANLoss
from utils import get_default_argparser
from utils import load_checkpoints, save_checkpoints

from preprocess import NikoPairedDataset, save_image

from torchvision import transforms

from PIL import Image


def main(args):
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # assign data loader
    train_data = NikoPairedDataset(transform=train_transform)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
    )
    val_loader = NikoPairedDataset(
        transform=val_transform,
        mode='val',
    )

    # assign model
    generator = VggUnet(resolution=512, norm='batch', dim=64).to(device)
    discriminator = PatchGAN(norm='batch', dim=64).to(device)

    # assign loss
    gan_loss = GANLoss(False).to(device)  # use MSE Loss
    l1_loss = nn.L1Loss().to(device)

    # assign optimizer
    optimG = optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))
    optimD = optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999))

    # load pretrained model
    if args.pretrainedG != '':
        load_checkpoints(args.pretrainedG, generator, optimG)
    if args.pretrainedD != '':
        load_checkpoints(args.pretrainedD, discriminator, optimD)

    def center_crop(tensor_image):
        _, _, h, w = tensor_image.shape
        tensor_image = tensor_image[:, :, h // 2 - 112:h // 2 + 112, w // 2 -
                                    112:w // 2 + 112]
        return tensor_image

    def train(last_iter):
        #  idx_range = range(0, len(train_data) - 1)
        for i, datas in enumerate(train_loader, last_iter + 1):
            # sample other
            #  idx = random.choice(idx_range)

            #  styleA, styleB = train_data[idx]
            imageA, imageB = datas
            if args.mode == 'B2A':
                # swap
                imageA, imageB = imageB, imageA
                #  styleA, styleB = styleB, styleA

            imageA = imageA.to(device)
            imageB = imageB.to(device)
            #  styleB = styleB.unsqueeze(0).to(device)
            fakeB, guideB1, guideB2 = generator(imageA, center_crop(imageB))

            # proceed Discriminator
            optimD.zero_grad()
            real_AB = torch.cat([imageA, imageB], 1)
            logit_real = discriminator(real_AB)
            d_loss_real = gan_loss(logit_real, True)

            fake_AB = torch.cat([imageA, fakeB], 1)
            logit_fake = discriminator(fake_AB.detach())
            d_loss_fake = gan_loss(logit_fake, False)
            d_loss = (d_loss_fake + d_loss_real) * 0.5
            d_loss.backward()
            optimD.step()

            # proceed Generator
            optimG.zero_grad()

            fake_AB = torch.cat([imageA, fakeB], 1)
            logit_fake = discriminator(fake_AB)
            g_loss_gan = gan_loss(logit_fake, True)

            g_loss_l1_g1 = l1_loss(guideB1, imageB) * args.alpha

            g_loss_l1_g2 = l1_loss(guideB2, imageB) * args.beta

            g_loss_l1 = (l1_loss(fakeB, imageB) + g_loss_l1_g1 +
                         g_loss_l1_g2) * args.lambd
            g_loss = g_loss_gan + g_loss_l1
            g_loss.backward()
            optimG.step()

            if args.verbose and i % args.print_every == 0:
                print(
                    'Iter %d: d_loss_real = %f, d_loss_fake = %f, g_loss = %f, l1_loss = %f, l1_g1_loss = %f, l1_g2_loss = %f'
                    % (i, d_loss_real, d_loss_fake, g_loss_gan, g_loss_l1,
                       g_loss_l1_g1, g_loss_l1_g2))

        return i

    def validate(epoch=0):
        length = len(val_loader)

        # sample 3 images
        idxs = random.sample(range(0, length - 1), 6)
        styles = idxs[3:]
        idxs = idxs[0:3]

        sample = Image.new('RGB', (4 * 512, 3 * 512))
        recover = transforms.ToPILImage()

        for i, (idx, style) in enumerate(zip(idxs, styles)):
            concat = Image.new('RGB', (4 * 512, 512))
            imageA, imageB = val_loader[idx]
            styleA, styleB = val_loader[style]

            if args.mode == 'B2A':
                # A is a sketch
                # B is a ground truth
                imageA, imageB = imageB, imageA
                styleA, styleB = styleB, styleA

            imageA = imageA.unsqueeze(0).to(device)
            styleB = styleB.unsqueeze(0).to(device)
            fakeB = generator(imageA, center_crop(styleB))[0].squeeze()
            styleB = styleB.squeeze()
            imageA = imageA.squeeze()

            imageA = ((imageA + 1) * 0.5).detach().cpu()
            imageB = ((imageB + 1) * 0.5).detach().cpu()
            styleB = ((styleB + 1) * 0.5).detach().cpu()
            fakeB = ((fakeB + 1) * 0.5).detach().cpu()

            imageA = recover(imageA)
            imageB = recover(imageB)
            styleB = recover(styleB)
            fakeB = recover(fakeB)

            concat.paste(imageA, (0, 0))
            concat.paste(styleB, (512, 0))
            concat.paste(fakeB, (2 * 512, 0))
            concat.paste(imageB, (3 * 512, 0))

            sample.paste(concat, (0, 0 + 512 * i))

        save_image(sample, 'vggunet_val_%03d' % epoch,
                   './data/pair_niko/result')

    if args.train:
        last_iter = -1

        for epoch in range(args.num_epochs):
            last_iter = train(last_iter)

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoints(
                    generator, 'VggUnetG', epoch, optimizer=optimG)
                save_checkpoints(
                    discriminator, 'VggUnetD', epoch, optimizer=optimD)
            validate(epoch)
            print('Epoch %d finished' % epoch)

    else:
        validate()


if __name__ == "__main__":
    parser = get_default_argparser()
    parser.add_argument(
        '--use-mse',
        help='set whether to use mean square loss in gan loss',
        action='store_true',
    )
    parser.add_argument(
        '--lambd',
        help='set l1 loss weight',
        metavar='',
        type=float,
        default=100.)
    parser.add_argument(
        '--alpha',
        help='set l1 loss alpha weight',
        metavar='',
        type=float,
        default=0.3)
    parser.add_argument(
        '--beta',
        help='set l1 loss beta weight',
        metavar='',
        type=float,
        default=0.9)
    parser.add_argument(
        '--mode', help='set mapping mode', metavar='', type=str, default='A2B')
    parser.add_argument(
        '--pretrainedG',
        help='set pretrained generator',
        metavar='',
        type=str,
        default='')
    parser.add_argument(
        '--pretrainedD',
        help='set pretrained discriminator',
        metavar='',
        type=str,
        default='')

    main(parser.parse_args())
