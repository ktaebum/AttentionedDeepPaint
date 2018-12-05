import argparse


def get_default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--print-every',
        help='print every specific number of iteration',
        metavar='',
        type=int,
        default=300)
    parser.add_argument(
        '--batch-size',
        help='set number of batch size',
        metavar='',
        type=int,
        default=4)
    parser.add_argument(
        '--last-epoch',
        help='set last epoch of training (default: 0)',
        metavar='',
        type=int,
        default=0)
    parser.add_argument(
        '--num-epochs',
        help='set number of total epochs',
        metavar='',
        type=int,
        default=20)
    parser.add_argument(
        '--learning-rate',
        help='set training learning rate',
        metavar='',
        type=float,
        default=0.0002)
    parser.add_argument(
        '--beta1',
        help='set beta1 value in adam optimizer',
        metavar='',
        type=float,
        default=0.5)
    parser.add_argument(
        '--verbose',
        help='set whether verbose train or not',
        action='store_true')
    parser.add_argument(
        '--train',
        help='set whether run in train mode or not',
        action='store_true')
    parser.add_argument(
        '--weight-decay',
        help='set training weight decay rate',
        metavar='',
        type=float,
        default=0.)
    parser.add_argument(
        '--dropout',
        help='set dropout rate ',
        metavar='',
        type=float,
        default=0.5)
    parser.add_argument(
        '--save-every',
        help='set epoch frequency of saving model',
        metavar='',
        type=int,
        default=0)
    parser.add_argument(
        '--sample',
        help='set number of sample images in validation',
        metavar='',
        type=int,
        default=3)
    parser.add_argument(
        '--save-name',
        help='set model save name (effective if and only if save_every > 0)',
        metavar='',
        type=str,
        default='')
    parser.add_argument(
        '--model',
        help='set model to use (\'vggunet\')',
        metavar='',
        type=str,
        default='vggunet')
    parser.add_argument(
        '--no-mse',
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
        '--mode',
        help='set image translation mapping mode',
        metavar='',
        type=str,
        default='B2A')
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
    return parser
