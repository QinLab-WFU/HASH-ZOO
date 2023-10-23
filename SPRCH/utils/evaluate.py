import numpy as np
import torch
import torch.backends.cudnn as cudnn


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True


def feed_random_seed(seed=59495):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_ratio(dataloader):
    pos_ratio = []
    neg_ratio = []
    for i, (image, label, _) in enumerate(dataloader):
        label = label.cuda()
        pos_mask = (torch.mm(label.float(), label.float().T) > 0).float()
        neg_mask = 1 - pos_mask
        pos_mask2 = label
        neg_mask2 = 1 - label
        pos_ratio.append(pos_mask.sum(1) / pos_mask2.sum(1))
        neg_ratio.append(neg_mask.sum(1) / neg_mask2.sum(1))
    return torch.cat(pos_ratio), torch.cat(neg_ratio)
