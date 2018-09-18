"""
Loss functions module for pytorch
"""

import torch
import torch.nn as nn


def mse_gan_loss(logit_real, logit_fake):
    loss_function = nn.MSELoss()
    target_ones_real = torch.ones_like(logit_real)
    target_ones_fake = torch.ones_like(logit_fake)
    target_zeros = torch.zeros_like(logit_fake)

    loss_G = loss_function(logit_fake.detach(), target_ones_fake)
    loss_D = loss_function(logit_real, target_ones_real) + loss_function(
        logit_fake, target_zeros)

    return loss_G, loss_D
