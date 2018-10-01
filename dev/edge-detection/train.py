# main program to train hed model
# Author: Nishanth
# Date: 2017/10/19

# import torch libraries
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# import utility functions
from model import *
from trainer import Trainer
from dataproc import TrainDataset

# fix random seed
rng = np.random.RandomState(37148)

# GPU ID
gpuID = 0

# batch size
nBatch = 1

# load the images dataset
dataRoot = 'data/HED-BSDS/'
modelPath = 'model/vgg16.pth'
valPath = dataRoot+'val_pair.lst'
trainPath = dataRoot+'train_pair.lst'

# create data loaders from dataset
std=[0.229, 0.224, 0.225]
mean=[0.485, 0.456, 0.406]

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
targetTransform = transforms.Compose([
                transforms.ToTensor()
            ])

valDataset = TrainDataset(valPath, dataRoot, 
                          transform, targetTransform)
trainDataset = TrainDataset(trainPath, dataRoot, 
                            transform, targetTransform)

valDataloader = DataLoader(valDataset, shuffle=False)
trainDataloader = DataLoader(trainDataset, shuffle=True)

# initialize the network
net = HED()
net.apply(weights_init)

pretrained_dict = torch.load(modelPath)
pretrained_dict = convert_vgg(pretrained_dict)

model_dict = net.state_dict()
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

net.cuda(gpuID)

# define the optimizer
lr = 1e-4
lrDecay = 1e-1
lrDecayEpoch = {3,5,8,10,12}

fuse_params = list(map(id, net.fuse.parameters()))
conv5_params = list(map(id, net.conv5.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params+fuse_params,
                     net.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': net.conv5.parameters(), 'lr': lr * 100},
            {'params': net.fuse.parameters(), 'lr': lr * 0.001}
            ], lr=lr, momentum=0.9)

# initialize trainer class
trainer = Trainer(net, optimizer, trainDataloader, valDataloader, 
                  nBatch=nBatch, maxEpochs=15, cuda=True, gpuID=gpuID,
                  lrDecayEpochs=lrDecayEpoch)

# train the network
trainer.train()
