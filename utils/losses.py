"""
Module for various loss functions
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, mse=True):
        super(GANLoss, self).__init__()
        if mse:
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.BCELoss()

    def forward(self, logit, is_real):
        if is_real:
            target = torch.ones_like(logit)
        else:
            target = torch.zeros_like(logit)

        return self.loss_function(logit, target)
