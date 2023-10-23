import random

import numpy as np
import torch
from torch.optim import SGD, Adam, lr_scheduler
from torchvision import transforms

import models
from OrthoHash.models.alexnet import AlexNet


def R(config):
    r = {
        'cifar': -1,  # mAP@all
        'nuswide': 5000,
        'flickr': -1,
        'coco': -1
    }[config['dataset']]

    return r


def arch(config, **kwargs):
    print('models.network_names', config['arch'])
    if config['arch'] == "alexnet":
        net = AlexNet(**config['arch_kwargs'], **kwargs)
    else:
        raise ValueError(f'Invalid Arch: {config["arch"]}')

    return net


def optimizer(config, params):
    o_type = config['optim']
    kwargs = config['optim_kwargs']

    if o_type == 'sgd':
        o = SGD(params,
                lr=kwargs['lr'],
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0005),
                nesterov=kwargs.get('nesterov', False))
    else:  # adam
        o = Adam(params,
                 lr=kwargs['lr'],
                 betas=kwargs.get('betas', (0.9, 0.999)),
                 weight_decay=kwargs.get('weight_decay', 0))

    return o


def scheduler(config, optimizer):
    s_type = config['scheduler']
    kwargs = config['scheduler_kwargs']

    if s_type == 'step':
        return lr_scheduler.StepLR(optimizer,
                                   kwargs['step_size'],
                                   kwargs['gamma'])
    elif s_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer,
                                        [int(float(m) * int(config['epochs'])) for m in
                                         kwargs['milestones'].split(',')],
                                        kwargs['gamma'])
    else:
        raise Exception('Scheduler not supported yet: ' + s_type)


def compose_transform(mode='train', resize=0, crop=0, norm=0,
                      augmentations=None):
    """

    :param mode:
    :param resize:
    :param crop:
    :param norm:
    :param augmentations:
    :return:
    if train:
      Resize (optional, usually done in Augmentations)
      Augmentations
      ToTensor
      Normalize

    if test:
      Resize
      CenterCrop
      ToTensor
      Normalize
    """
    # norm = 0, 0 to 1
    # norm = 1, -1 to 1
    # norm = 2, standardization
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]

    compose = []

    if resize != 0:
        compose.append(transforms.Resize(resize))

    if mode == 'train' and augmentations is not None:
        compose += augmentations

    if mode == 'test' and crop != 0 and resize != crop:
        compose.append(transforms.CenterCrop(crop))

    compose.append(transforms.ToTensor())

    if norm != 0:
        compose.append(transforms.Normalize(mean, std))

    return transforms.Compose(compose)


def build_trans(config, transform_mode):
    dataset_name = config['dataset']

    resize = config['dataset_kwargs'].get('resize', 0)
    crop = config['dataset_kwargs'].get('crop', 0)
    norm = config['dataset_kwargs'].get('norm', 2)

    if dataset_name in ['nuswide', 'flickr', 'coco']:
        if transform_mode == 'train':
            trans = compose_transform('train', 0, crop, 2, [
                transforms.Resize(resize),
                transforms.RandomCrop(crop),
                transforms.RandomHorizontalFlip()
            ])
        else:
            trans = compose_transform('test', resize, crop, 2)

    else:  # cifar
        resizec = 0 if resize == 32 else resize
        cropc = 0 if crop == 32 else crop

        if transform_mode == 'train':
            trans = compose_transform('train', resizec, 0, norm, [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])
        else:
            trans = compose_transform('test', resizec, cropc, norm)

    return trans


def seeding(seed):
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
