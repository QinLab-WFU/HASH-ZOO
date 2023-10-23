import argparse
import json
import os

import torch
from loguru import logger

from OrthoHash import configs
from OrthoHash.configs import build_trans
from OrthoHash.scripts import train_hashing
from _data import build_loader


def get_config():
    parser = argparse.ArgumentParser(description='OrthoHash')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--arch', default='alexnet', choices=['alexnet'], help='backbone name')

    # loss related
    parser.add_argument('--scale', default=8, type=float, help='scale for cossim')
    parser.add_argument('--margin', default=0.2, type=float, help='ortho margin')
    parser.add_argument('--margin-type', default='cos', choices=['cos', 'arc'], help='margin type')
    parser.add_argument('--ce', default=1.0, type=float, help='classification scale')
    parser.add_argument('--quan', default=0.0, type=float, help='quantization loss scale')
    parser.add_argument('--quan-type', default='cs', choices=['cs', 'l1', 'l2'], help='quantization types')
    parser.add_argument('--multiclass-loss', default='label_smoothing',
                        choices=['bce', 'imbalance', 'label_smoothing'], help='multiclass loss types')

    # codebook generation
    parser.add_argument('--codebook-method', default='B', choices=['N', 'B', 'O'], help='N = sign of gaussian; '
                                                                                        'B = bernoulli; '
                                                                                        'O = optimize')

    parser.add_argument('--seed', default=torch.randint(100000, size=()).item(), help='seed number; default: random')

    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    config = {
        'arch': args.arch,
        'arch_kwargs': {
            'nbit': 0,  # will be updated below
            'nclass': 0,  # will be updated below
            'pretrained': True,
            'freeze_weight': False,
        },
        'batch_size': args.bs,
        'dataset': '',  # will be updated below
        'dataset_kwargs': {
            'resize': 256,  # will be updated below
            'crop': 224,
            'norm': 2,
            'root': '/home/sxz/Downloads/datasets',
            # 'separate_multiclass': False,  # 是否把多标签分成多个图片和单标签
        },
        'optim': 'adam',
        'optim_kwargs': {
            'lr': args.lr,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'nesterov': False,
            'betas': (0.9, 0.999)
        },
        'epochs': args.epochs,
        'scheduler': 'step',
        'scheduler_kwargs': {
            'step_size': int(args.epochs * 0.8),
            'gamma': 0.1,
            'milestones': '0.5,0.75'
        },
        'eval_interval': 5,
        'tag': 'orthohash',
        'seed': args.seed,

        'codebook_generation': args.codebook_method,

        # loss_param
        'ce': args.ce,
        's': args.scale,
        'm': args.margin,
        'm_type': args.margin_type,
        'quan': args.quan,
        'quan_type': args.quan_type,
        'multiclass_loss': args.multiclass_loss,
        'device': args.device
    }
    return config


def build_loader_local(config):
    trans_train = build_trans(config, "train")
    trans_test = build_trans(config, "test")

    train_loader, test_loader, database_loader, topk, num_classes = build_loader(config['dataset_kwargs']['root'],
                                                                                 config['dataset'],
                                                                                 config['batch_size'], os.cpu_count(),
                                                                                 trans_train,
                                                                                 trans_test,
                                                                                 True,
                                                                                 False)

    config['R'] = topk
    config['arch_kwargs']['nclass'] = num_classes
    return train_loader, test_loader, database_loader


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    config = get_config()
    configs.seeding(config['seed'])

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        logger.info(f'processing dataset: {dataset}')
        config['dataset'] = dataset
        config['multiclass'] = dataset != 'cifar'
        config['dataset_kwargs']['resize'] = 256 if dataset != 'cifar' else 224

        train_loader, test_loader, db_loader = build_loader_local(config)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')
            config["arch_kwargs"]["nbit"] = hash_bit

            config["save_dir"] = f"./output/{config['arch']}/{dataset}/{hash_bit}"
            os.makedirs(config["save_dir"], exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{config["save_dir"]}/train.log', rotation="500 MB", level="INFO")

            with open(f'{config["save_dir"]}/config.json', 'w+') as f:
                json.dump(config, f, indent=4, sort_keys=True)

            best_epoch, best_map = train_hashing.main(config, train_loader, test_loader, db_loader, logger)

            rst.append(
                {
                    "dataset": dataset,
                    "hash_bit": hash_bit,
                    "best_epoch": best_epoch,
                    "best_map": best_map,
                }
            )

    for x in rst:
        print(
            f"[dataset:{x['dataset']}][hash-bit:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )
