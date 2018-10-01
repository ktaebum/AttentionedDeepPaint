# trainer.py: Utility function to train HED model
# Author: Nishanth Koganti
# Date: 2017/10/20

# Source: https://github.com/xlliu7/hed.pytorch/blob/master/trainer.py

# Issues:
#

# import libraries
import math
import os, time
import numpy as np
from PIL import Image
import os.path as osp

# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# utility class for training HED model
class Trainer(object):
    # init function for class
    def __init__(self,
                 generator,
                 optimizerG,
                 trainDataloader,
                 valDataloader,
                 nBatch=10,
                 out='train',
                 maxEpochs=1,
                 cuda=True,
                 gpuID=0,
                 lrDecayEpochs={}):

        # set the GPU flag
        self.cuda = cuda
        self.gpuID = gpuID

        # define an optimizer
        self.optimG = optimizerG

        # set the network
        self.generator = generator

        # set the data loaders
        self.valDataloader = valDataloader
        self.trainDataloader = trainDataloader

        # set output directory
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        # set training parameters
        self.epoch = 0
        self.nBatch = nBatch
        self.nepochs = maxEpochs
        self.lrDecayEpochs = lrDecayEpochs

        self.gamma = 0.1
        self.valInterval = 500
        self.dispInterval = 100
        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        # function to train network
        for epoch in range(self.epoch, self.nepochs):
            # set function to training mode
            self.generator.train()

            # initialize gradients
            self.optimG.zero_grad()

            # adjust hed learning rate
            if epoch in self.lrDecayEpochs:
                self.adjustLR()

            # train the network
            losses = []
            lossAcc = 0.0
            for i, sample in enumerate(self.trainDataloader, 0):
                # get the training batch
                data, target = sample

                if self.cuda:
                    data, target = data.cuda(self.gpuID), target.cuda(
                        self.gpuID)
                data, target = Variable(data), Variable(target)

                # generator forward
                tar = target
                d1, d2, d3, d4, d5, d6 = self.generator(data)

                # compute loss for batch
                loss1 = self.bce2d(d1, tar)
                loss2 = self.bce2d(d2, tar)
                loss3 = self.bce2d(d3, tar)
                loss4 = self.bce2d(d4, tar)
                loss5 = self.bce2d(d5, tar)
                loss6 = self.bce2d(d6, tar)

                # all components have equal weightage
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')

                losses.append(loss)
                lossAcc += loss.data[0]

                # perform backpropogation and update network
                if i % self.nBatch == 0:
                    bLoss = sum(losses)

                    bLoss.backward()
                    self.optimG.step()
                    self.optimG.zero_grad()

                    losses = []

                # visualize the loss
                if (i + 1) % self.dispInterval == 0:
                    timestr = time.strftime(self.timeformat, time.localtime())
                    print("%s epoch: %d iter:%d loss:%.6f" %
                          (timestr, epoch + 1, i + 1,
                           lossAcc / self.dispInterval))
                    lossAcc = 0.0

                # perform validation every 500 iters
                if (i + 1) % self.valInterval == 0:
                    self.val(epoch + 1)

            # save model after every epoch
            torch.save(self.generator.state_dict(), '%s/HED.pth' % (self.out))

    def val(self, epoch):
        # eval model on validation set
        print('Evaluation:')

        # convert to test mode
        self.generator.eval()

        # save the results
        if os.path.exists(self.out + '/images') == False:
            os.mkdir(self.out + '/images')
        dirName = '%s/images' % (self.out)

        # perform test inference
        for i, sample in enumerate(self.valDataloader, 0):
            # get the test sample
            data, target = sample

            if self.cuda:
                data, target = data.cuda(self.gpuID), target.cuda(self.gpuID)
            data, target = Variable(data), Variable(target)

            # perform forward computation
            d1, d2, d3, d4, d5, d6 = self.generator.forward(data)

            # transform to grayscale images
            d1 = self.grayTrans(self.crop(d1))
            d2 = self.grayTrans(self.crop(d2))
            d3 = self.grayTrans(self.crop(d3))
            d4 = self.grayTrans(self.crop(d4))
            d5 = self.grayTrans(self.crop(d5))
            d6 = self.grayTrans(self.crop(d6))
            tar = self.grayTrans(self.crop(target))

            d1.save('%s/sample%d1.png' % (dirName, i))
            d2.save('%s/sample%d2.png' % (dirName, i))
            d3.save('%s/sample%d3.png' % (dirName, i))
            d4.save('%s/sample%d4.png' % (dirName, i))
            d5.save('%s/sample%d5.png' % (dirName, i))
            d6.save('%s/sample%d6.png' % (dirName, i))
            tar.save('%s/sample%dT.png' % (dirName, i))

        print('evaluate done')
        self.generator.train()

    # function to crop the padding pixels
    def crop(self, d):
        d_h, d_w = d.size()[2:4]
        g_h, g_w = d_h - 64, d_w - 64
        d1 = d[:, :,
               int(math.floor((d_h - g_h) /
                              2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
               int(math.floor((d_w - g_w) /
                              2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
        return d1

    def _assertNoGrad(self, variable):
        assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

    # binary cross entropy loss in 2D
    def bce2d(self, input, target):
        n, c, h, w = input.size()

        # assert(max(target) == 1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(
            1, -1)
        target_trans = target_t.clone()
        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy(
            log_p, target_t, weight, size_average=True)
        return loss

    def grayTrans(self, img):
        img = img.data.cpu().numpy()[0][0] * 255.0
        img = (img).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img

    # utility functions to set the learning rate
    def adjustLR(self):
        for param_group in self.optimG.param_groups:
            param_group['lr'] *= self.gamma
