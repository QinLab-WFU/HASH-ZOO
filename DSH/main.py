import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from timm.utils import AverageMeter
from torch.autograd import Variable
from torch.backends import cudnn

from _data import build_loader
from model import *
from _utils import prediction, mean_average_precision


def hashing_loss_buggy(b, cls, m, alpha):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    # y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1) # 输入分类时用这个
    y = (cls @ cls.T == 0).float().view(-1)  # 输入one-hot分类时用这个
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=2).view(-1)
    loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)
    loss = loss.mean() + alpha * (b.abs() - 1).abs().sum(dim=1).mean() * 2
    return loss


def hashing_loss_paper(b, cls, m, alpha):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    y = (cls @ cls.T == 0).float()
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=-1)
    loss1 = ((1 - y) / 2 * dist).sum()
    loss2 = (y / 2 * (m - dist).clamp(min=0)).sum()
    loss3 = alpha * (b.abs() - 1).abs().sum()
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3


def hashing_loss(b, cls, m, alpha):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    y = (cls @ cls.T == 0).float()
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=-1)
    loss1 = ((1 - y) / 2 * dist).mean()
    loss2 = (y / 2 * (m - dist).clamp(min=0)).mean()
    loss3 = alpha * (b.abs() - 1).abs().mean()
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3


def train(dataloader, net, optimizer, m, alpha):
    loss_meters = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    net.train()
    for i, (img, cls, _) in enumerate(dataloader):
        img, cls = [Variable(x.cuda()) for x in (img, cls)]

        net.zero_grad()
        b = net(img)
        losses = hashing_loss(b, cls, m, alpha)

        losses[0].backward()
        optimizer.step()
        # scheduler.step()
        for i, loss in enumerate(losses):
            loss_meters[i].update(loss.item())
    return loss_meters[0].avg, loss_meters[1].avg, loss_meters[2].avg, loss_meters[3].avg


def get_config():
    parser = argparse.ArgumentParser(description='train DSH')
    parser.add_argument('--dataset', default='cifar', help='name of dataset')
    parser.add_argument('--topk', default='-1', help='for calc map')
    parser.add_argument('--root', default='/home/sxz/Downloads/datasets', help='path to dataset')
    parser.add_argument('--weights', default='', help="path to weight (to continue training)")
    parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpointing after batches')

    parser.add_argument('--batch_size', type=int, default=200, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=0, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=12, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.01, help='weighting of regularizer')

    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate') # paper
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=4e-3, help='weight decay') # paper
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--model', default='alexnet', help='model name')
    return parser.parse_args()


def run(opt, train_loader, test_loader, database_loader, logger):
    # setup net
    if opt.model == 'alexnet':
        net = AlexNet(opt.binary_bits)
    elif opt.model == 'dsh':
        net = DSH(opt.binary_bits)
    else:
        raise NotImplementedError(f"unknown model: {opt.model}")

    resume_epoch = 0
    # print(net)
    if opt.weights:
        print(f'loading weight form {opt.weights}')
        resume_epoch = int(os.path.basename(opt.weights)[:-4])
        net.load_state_dict(torch.load(opt.weights, map_location=lambda storage, location: storage))

    net.cuda()

    # setup optimizer
    # paper's optimizer will cause exploding gradients in cifar 128bit
    # optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)  # paper
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.6)  # paper
    # optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)  # opt.lr = 0.001
    # optimizer = optim.Adam(net.parameters(), lr=4e-5, weight_decay=1e-5)  # opt.lr = 0.001
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    iter = 0
    best_iter = 0
    for epoch in range(resume_epoch, opt.niter):
        train_losses = train(train_loader, net, optimizer, 2 * opt.binary_bits, opt.alpha)
        iter += len(train_loader)
        logger.info(
            f"[Train][dataset:{opt.dataset}][bits:{opt.binary_bits}][epoch:{epoch}/{opt.niter - 1}][iters:{iter}][train_loss:{train_losses[0]}][loss1:{train_losses[1]}][loss2:{train_losses[2]}][loss3:{train_losses[3]}]")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.niter:
            # compute mAP by searching testset images from trainset
            qB, qL = prediction(net, test_loader)
            rB, rL = prediction(net, database_loader)
            # tB, tL = prediction(net, train_loader)
            map = mean_average_precision(qB, rB, qL, rL, opt.topk)
            # map2 = calc_map_k(qB, tB, qL, tL, opt.topk)
            logger.info(
                f"[Evaluation][dataset:{opt.dataset}][bits:{opt.binary_bits}][epoch:{epoch}/{opt.niter - 1}][iters:{iter}][best-mAP@{opt.topk}:{best_map:.7f}][mAP@{opt.topk}:{map:.7f}][count:{0 if map > best_map else (count + 1)}]")  # [mAP-train@{opt.topk}:{map2}]")
            # if (epoch + 1) < 500:
            # 没预训练，到一定训练量再判断收敛
            # continue
            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_iter = iter
                best_checkpoint = deepcopy(net.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
                    torch.save(best_checkpoint, f"{opt.outf}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{opt.outf}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_iter, best_map


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_loader_local(opt):
    train_loader, test_loader, database_loader, topk, _ = build_loader(opt.root,
                                                                       opt.dataset,
                                                                       opt.batch_size, 4)

    opt.topk = topk

    return train_loader, test_loader, database_loader


if __name__ == '__main__':
    opt = get_config()

    choose_gpu(opt.ngpu)
    feed_random_seed()

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    dummy_logger_id = None
    rst = []
    for dataset in ['cifar', 'nuswide', 'flickr', 'coco']:
        logger.info(f'processing dataset: {dataset}')
        opt.dataset = dataset

        train_loader, test_loader, database_loader = build_loader_local(opt)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')
            opt.binary_bits = hash_bit

            opt.outf = f"./output/{opt.model}/{dataset}/{hash_bit}"
            os.makedirs(opt.outf, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{opt.outf}/train.log', rotation="500 MB", level="INFO")

            with open(f'{opt.outf}/config.json', 'w+') as f:
                json.dump(vars(opt), f, indent=4, sort_keys=True)

            best_epoch, best_iter, best_map = run(opt, train_loader, test_loader, database_loader, logger)
            rst.append(
                {
                    "dataset": dataset,
                    "hash_bit": hash_bit,
                    "best_epoch": best_epoch,
                    "best_iter": best_iter,
                    "best_map": best_map,
                }
            )

    for x in rst:
        print(
            f"[dataset:{x['dataset']}][hash-bit:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-iter:{x['best_iter']}][best-mAP:{x['best_map']:.3f}]"
        )
