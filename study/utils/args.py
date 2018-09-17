import argparse


def get_default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--print-every',
        help='print every specific number of iteration',
        metavar='',
        type=int,
        default=100)
    parser.add_argument(
        '--batch-size',
        help='set number of batch size',
        metavar='',
        type=int,
        default=64)
    parser.add_argument(
        '--num-epochs',
        help='set number of total epochs',
        metavar='',
        type=int,
        default=5)
    parser.add_argument(
        '--learning-rate',
        help='set training learning rate',
        metavar='',
        type=float,
        default=1e-3)
    parser.add_argument(
        '--verbose',
        help='set whether verbose train or not',
        action='store_true')
    parser.add_argument(
        '--train',
        help='set whether run in train mode or not',
        action='store_true')

    return parser
