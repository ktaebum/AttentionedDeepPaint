import torch
import torch.nn as nn
import torch.optim as optim

import random

from torchvision import transforms

from PIL import Image

from trainer.trainer import ModelTrainer

from models import PatchGAN, ResUnet

from utils import GANLoss
from utils import load_checkpoints, save_checkpoints
from utils import AverageTracker

from preprocess import centor_crop_tensor, re_scale
from preprocess import save_image


class ResUnetTrainer(ModelTrainer):
    def __init__(self, *args):
        super(ResUnetTrainer, self).__init__(*args)

        # build model
        self.resolution = self.args.resolution
        self.generator = ResUnet(
            resblock=self.args.resblock, norm=self.args.norm).to(self.device)
        self.discriminator = PatchGAN(
            dim=self.args.dim, norm=self.args.norm,
            sigmoid=self.args.no_mse).to(self.device)

        # set optimizers
        self.optimizers = self._set_optimizers()

        # set loss functions
        self.losses = self._set_losses()

        # load pretrained model
        if self.args.pretrainedG != '':
            load_checkpoints(self.args.pretrainedG, self.generator,
                             self.optimizers['G'])
        if self.args.pretrainedD != '':
            load_checkpoints(self.args.pretrainedD, self.discriminator,
                             self.optimizers['D'])

        if self.device.type == 'cuda':
            # enable parallel computation
            self.generator = nn.DataParallel(
                self.generator, device_ids=range(torch.cuda.device_count()))
            self.discriminator = nn.DataParallel(
                self.discriminator,
                device_ids=range(torch.cuda.device_count()))

        # loss values for tracking
        self.loss_G_gan = AverageTracker('loss_G_gan')
        self.loss_G_l1 = AverageTracker('loss_G_l1')
        self.loss_G_guide1 = AverageTracker('loss_G_guide1')
        self.loss_G_guide2 = AverageTracker('loss_G_guide2')
        self.loss_D_real = AverageTracker('loss_D_real')
        self.loss_D_fake = AverageTracker('loss_D_fake')

        # image value
        self.imageA = None
        self.imageB = None
        self.fakeB = None
        self.guide1 = None
        self.guide2 = None

    def train(self, last_iteration):
        """
        Run single epoch
        """
        average_trackers = [
            self.loss_G_gan, self.loss_D_fake, self.loss_D_real,
            self.loss_G_guide1, self.loss_G_guide2, self.loss_G_l1
        ]
        for tracker in average_trackers:
            tracker.initialize()
        for i, datas in enumerate(self.data_loader, last_iteration):
            imageA, imageB = datas
            if self.args.mode == 'B2A':
                # swap
                imageA, imageB = imageB, imageA

            self.imageA = imageA.to(self.device)
            self.imageB = imageB.to(self.device)

            # run forward propagation
            self.fakeB, self.guide1, self.guide2 = self.generator(
                imageA, centor_crop_tensor(imageB))

            self._update_discriminator()
            self._update_generator()

            if self.args.verbose and i % self.args.print_every == 0:
                print(
                    '%s = %f, %s = %f, %s = %f, %s = %f, %s = %f, %s = %f' % (
                        self.loss_D_real.name,
                        self.loss_D_real(),
                        self.loss_D_fake.name,
                        self.loss_D_fake(),
                        self.loss_G_gan.name,
                        self.loss_G_gan(),
                        self.loss_G_l1.name,
                        self.loss_G_l1(),
                        self.loss_G_guide1.name,
                        self.loss_G_guide1(),
                        self.loss_G_guide2.name,
                        self.loss_G_guide2(),
                    ))

        return i

    def validate(self, dataset, epoch, samples=3):
        length = len(dataset)

        # sample images
        idxs = random.sample(range(0, length - 1), samples * 2)
        styles = idxs[samples:]
        targets = idxs[0:samples]

        result = Image.new('RGB',
                           (4 * self.resolution, samples * self.resolution))

        toPIL = transforms.ToPILImage()

        for i, (target, style) in enumerate(zip(targets, styles)):
            sub_result = Image.new('RGB',
                                   (4 * self.resolution, self.resolution))
            imageA, imageB = dataset[target]
            styleA, styleB = dataset[style]

            if self.args.mode == 'B2A':
                imageA, imageB = imageB, imageA
                styleA, styleB = styleB, styleA

            imageA = imageA.unsqueeze(0).to(self.device)
            styleB = styleB.unsqueeze(0).to(self.device)
            fakeB, _, _ = self.generator(imageA, centor_crop_tensor(styleB))

            styleB = styleB.squeeze()
            fakeB = fakeB.squeeze()
            imageA = imageA.squeeze()

            imageA = toPIL(re_scale(imageA).detach().cpu())
            imageB = toPIL(re_scale(imageB).detach().cpu())
            styleB = toPIL(re_scale(styleB).detach().cpu())
            fakeB = toPIL(re_scale(fakeB).detach().cpu())

            sub_result.paste(imageA, (0, 0))
            sub_result.paste(styleB, (512, 0))
            sub_result.paste(fakeB, (2 * 512, 0))
            sub_result.paste(imageB, (3 * 512, 0))

            result.paste(sub_result, (0, 0 + self.resolution * i))

        name = 'resunet' if self.args.resblock else 'vgg'

        save_image(result, '%s_val_%03d' % (name, epoch),
                   './data/pair_niko/result')

    def test(self):
        raise NotImplementedError

    def save_model(self, name, epoch):
        save_checkpoints(
            self.generator,
            name + 'G',
            epoch,
            optimizer=self.optimizers['G'],
        )
        save_checkpoints(
            self.discriminator,
            name + 'D',
            epoch,
            optimizer=self.optimizers['D'])

    def _set_optimizers(self):
        optimG = optim.Adam(
            self.generator.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, 0.999))
        optimD = optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, 0.999))

        return {'G': optimG, 'D': optimD}

    def _set_losses(self):
        gan_loss = GANLoss(not self.args.no_mse).to(self.device)
        l1_loss = nn.L1Loss().to(self.device)

        return {'GAN': gan_loss, 'L1': l1_loss}

    def _update_generator(self):
        optimG = self.optimizers['G']
        gan_loss = self.losses['GAN']
        l1_loss = self.losses['L1']
        batch_size = self.imageA.shape[0]

        optimG.zero_grad()
        fake_AB = torch.cat([self.imageA, self.fakeB], 1)
        logit_fake = self.discriminator(fake_AB)
        loss_G_gan = gan_loss(logit_fake, True)

        loss_G_guide1 = l1_loss(self.guide1, self.imageB) * self.args.alpha
        loss_G_guide2 = l1_loss(self.guide2, self.imageB) * self.args.beta
        loss_G_l1 = (l1_loss(self.fakeB, self.imageB) + loss_G_guide1 +
                     loss_G_guide2) * self.args.lambd

        self.loss_G_gan.update(loss_G_gan.item(), batch_size)
        self.loss_G_guide1.update(loss_G_guide1.item(), batch_size)
        self.loss_G_guide2.update(loss_G_guide2.item(), batch_size)
        self.loss_G_l1.update(loss_G_l1.item(), batch_size)

        loss_G = loss_G_gan + loss_G_l1

        loss_G.backward()
        optimG.step()

    def _update_discriminator(self):
        optimD = self.optimizers['D']
        gan_loss = self.losses['GAN']
        batch_size = self.imageA.shape[0]

        optimD.zero_grad()

        # for real image
        real_AB = torch.cat([self.imageA, self.imageB], 1)
        logit_real = self.discriminator(real_AB)
        loss_D_real = gan_loss(logit_real, True)
        self.loss_D_real.update(loss_D_real.item(), batch_size)

        # for fake image
        fake_AB = torch.cat([self.imageA, self.fakeB], 1)
        logit_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = gan_loss(logit_fake, False)
        self.loss_D_fake.update(loss_D_fake.item(), batch_size)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimD.step()
