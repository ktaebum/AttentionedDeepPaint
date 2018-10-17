"""
main script for overall model training
"""

import torch

from torch.utils.data import DataLoader

from trainer import VggUnetTrainer, ResGenTrainer, ResUnetTrainer
from trainer import Style2PaintTrainer

from utils import get_default_argparser

from preprocess import NikoPairedDataset

from torchvision import transforms

TRIANER_MAP = {
    'vggunet': VggUnetTrainer,
    'resgen': ResGenTrainer,
    'resunet': ResUnetTrainer,
    'style2paint': Style2PaintTrainer
}


def main(args):
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
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

    val_data = NikoPairedDataset(
        transform=val_transform,
        mode='val',
    )

    trainer = TRIANER_MAP.get(args.model, None)
    if trainer is None:
        raise KeyError('Non supporting model')

    trainer = trainer(args, train_loader, device)

    if args.train:
        last_iter = -1

        for epoch in range(args.last_epoch + 1,
                           args.last_epoch + 1 + args.num_epochs):
            last_iter = trainer.train(last_iter)

            if args.save_every > 0 and epoch % args.save_every == 0:
                trainer.save_model(args.model, epoch)

            trainer.validate(val_data, epoch, args.sample)
            print('Epoch %d finished' % epoch)

    else:
        trainer.validate(val_data, 0, args.sample)


if __name__ == "__main__":
    parser = get_default_argparser()
    main(parser.parse_args())
