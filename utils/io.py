import os
import torch
import torch.nn as nn

__save_path__ = './checkpoints'


def save_checkpoints(model,
                     save_name=None,
                     epoch=None,
                     evaluation=None,
                     optimizer=None):

    if not os.path.exists(__save_path__):
        os.mkdir(__save_path__)

    is_parallel = isinstance(model, nn.DataParallel)
    model_state = model.module.state_dict(
    ) if is_parallel else model.state_dict()

    save_state = {'model_state': model_state}

    if epoch is not None:
        save_state['last_epoch'] = epoch

    if evaluation is not None:
        save_state['evaluation'] = evaluation

    if optimizer is not None:
        save_state['optimizer'] = optimizer.state_dict()

    if save_name is None:
        if is_parallel:
            save_name = model.module.__class__.__name__
        else:
            save_name = model.__class__.__name__

    save_name = os.path.join(__save_path__,
                             '%s_%03d.pth.tar' % (save_name, epoch))

    torch.save(save_state, save_name)


def load_checkpoints(checkpoint, model, optimizer=None, device_type='cuda'):
    checkpoint = torch.load(
        os.path.join(__save_path__, checkpoint), map_location=device_type)

    last_epoch = checkpoint.get('last_epoch', 0)
    model_state = checkpoint.get('model_state', None)
    optim_state = checkpoint.get('optimizer', None)
    evaluation = checkpoint.get('evaluation', None)

    model.load_state_dict(model_state)
    if optimizer is not None and optim_state is not None:
        optimizer.load_state_dict(optim_state)

    return last_epoch, evaluation
