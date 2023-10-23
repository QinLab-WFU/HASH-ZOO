# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from timm.data import Mixup
from timm.utils import AverageMeter

from SWTH.config import get_config
from SWTH.data.build import build_transform
from SWTH.lr_scheduler import build_scheduler
from SWTH.models import build_model
from SWTH.my_loss import My_Loss
from SWTH.optimizer import build_optimizer
from SWTH.utils import get_grad_norm, auto_resume_helper, load_pretrained
from _data import build_loader
from _utils import prediction, mean_average_precision

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='./configs/swin_config.yaml', metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='/home/sxz/Downloads/datasets', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default='./pretrained/swin_tiny_patch4_window7_224.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=2, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='../output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, train_loader, test_loader, database_loader, mixup_fn, logger):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    # logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    criterion = My_Loss(config.MODEL.NUM_CLASSES, config.MODEL.hash_length, mixup_fn, config.MODEL.LABEL_SMOOTHING,
                        config.MODEL.alph_param, config.MODEL.beta_param, config.MODEL.gamm_param)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)

    if config.THROUGHPUT_MODE:
        throughput(test_loader, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    best_map = .0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger)
        # if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
        # save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.TRAIN.EPOCHS:
            map = test(config, model, test_loader, database_loader)
            logger.info(
                f"[Evaluation][dataset:{config.DATA.DATASET}][bits:{config.MODEL.hash_length}][epoch:{epoch}/{config.TRAIN.EPOCHS - 1}][best-mAP@{config.DATA.TOP_K}:{best_map:.7f}][mAP@{config.DATA.TOP_K}:{map:.7f}][count:{0 if map > best_map else (count + 1)}]")
            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_checkpoint = deepcopy(model.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP@{config.DATA.TOP_K}: {best_map}, best epoch: {best_epoch}")
                    torch.save(best_checkpoint, f"{config.OUTPUT}/e{best_epoch}_{best_map:.3f}.pkl")
                    break

    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{config.OUTPUT}/e{best_epoch}_{best_map:.3f}.pkl")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    return best_epoch, best_map


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hash_loss_meter = AverageMeter()
    quanti_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets, _) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        hash_out, cls_out = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            hash_loss, quanti_loss, cls_loss, loss = criterion(hash_out, cls_out, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            hash_loss, quanti_loss, cls_loss, loss = criterion(hash_out, cls_out, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        hash_loss_meter.update(hash_loss.item(), targets.size(0))
        quanti_loss_meter.update(quanti_loss.item(), targets.size(0))
        cls_loss_meter.update(cls_loss.item(), targets.size(0))

        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS - 1}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'hash_loss {hash_loss_meter.val:.4f} ({hash_loss_meter.avg:.4f})\t'
                f'quanti_loss {quanti_loss_meter.val:.4f} ({quanti_loss_meter.avg:.4f})\t'
                f'cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def test(config, model, test_loader, db_loader):
    qB, qL = prediction(model, test_loader, 0)
    rB, rL = prediction(model, db_loader, 0)
    map = mean_average_precision(qB, rB, qL, rL, config.DATA.TOP_K)
    return map


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def build_loader_local(config):
    trans_train = build_transform(True, config)
    trans_test = build_transform(False, config)

    train_loader, test_loader, database_loader, topk, num_classes = build_loader(config.DATA.DATA_PATH,
                                                                                 config.DATA.DATASET,
                                                                                 config.DATA.BATCH_SIZE,
                                                                                 config.DATA.NUM_WORKERS,
                                                                                 trans_train,
                                                                                 trans_test,
                                                                                 True,
                                                                                 False,
                                                                                 config.DATA.PIN_MEMORY)

    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.DATA.TOP_K = topk
    config.freeze()

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return train_loader, test_loader, database_loader, mixup_fn


if __name__ == '__main__':

    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    dummy_logger_id = None
    rst = []
    for dataset in ['cifar', 'nuswide', 'flickr', 'coco']:
        logger.info(f'processing dataset: {dataset}')

        config.defrost()
        config.DATA.DATASET = dataset
        config.freeze()

        train_loader, test_loader, database_loader, mixup_fn = build_loader_local(config)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')

            config.defrost()
            config.MODEL.hash_length = hash_bit
            config.OUTPUT = f"./output/{config.MODEL.TYPE}/{config.DATA.DATASET}/{config.MODEL.hash_length}"
            config.freeze()

            os.makedirs(config.OUTPUT, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{config.OUTPUT}/train.log', rotation="500 MB", level="INFO")

            with open(f'{config.OUTPUT}/config.json', 'w+') as f:
                json.dump(config, f, indent=4, sort_keys=True)

            best_epoch, best_map = main(config, train_loader, test_loader, database_loader, mixup_fn, logger)
            rst.append(
                {
                    "dataset": dataset,
                    "hash_bit": hash_bit,
                    "best_epoch": best_epoch,
                    "best_map": best_map,
                }
            )

    for x in rst:
        logger.info(
            f"[dataset:{x['dataset']}][hash-bit:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )
