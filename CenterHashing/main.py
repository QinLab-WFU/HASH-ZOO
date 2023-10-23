import json
import os
import random

import numpy as np
import torch
from loguru import logger
from torch import optim
from torchvision import transforms

from CenterHashing.scripts.train_CUB import train_val
from _data import build_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_config():
    config = {
        # "remarks": "OurLossWithPair",
        "seed": 60,
        "m": 16,
        "alpha": 1,
        "beta": 1.0,
        "beta2": 0.01,
        "mome": 0.9,
        "epoch_change": 9,
        "sigma": 1.0,
        "gamma": 20.0,
        "lambda": 0.0001,
        "mu": 1,
        "nu": 1,
        "eta": 55,
        "dcc_iter": 10,
        "optimizer": {
            "type": "optim.RMSprop",
            # "epoch_lr_decrease": 30,
            "optim_param": {
                "lr": 1e-5,
                "weight_decay": 1e-5,
                # "momentum": 0.9
                # "betas": (0.9, 0.999)
            },
        },
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": "moco",
        "dataset": "coco",  # flickr cifar nuswide
        "without_BN": False,
        "epoch": 1000,
        "test_map": 5,
        "stop_iter": 10,
        "n_gpu": torch.cuda.device_count(),
        # "bit_list": [16, 32, 64, 128],
        "max_norm": 5.0,
        "T": 1e-3,
        "label_size": 100,
        "update_center": False,
    }
    return config


def image_transform(resize_size, crop_size, is_train):
    if is_train:
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose(
        [transforms.Resize(resize_size)]
        + step
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loader_local(config):
    trans_test = image_transform(
        config["resize_size"], config["crop_size"], False
    )
    trans_train = image_transform(
        config["resize_size"], config["crop_size"], True
    )

    train_loader, test_loader, database_loader, topk, num_classes = build_loader("/home/sxz/Downloads/datasets",
                                                                                 config["dataset"],
                                                                                 config["batch_size"], 4,
                                                                                 trans_train,
                                                                                 trans_test)

    config["topk"] = topk
    config["n_class"] = num_classes

    return train_loader, test_loader, database_loader


if __name__ == "__main__":

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    config = get_config()

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        logger.info(f'processing dataset: {dataset}')
        config["dataset"] = dataset

        train_loader, test_loader, database_loader = build_loader_local(config)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')
            flag = True
            if hash_bit == 48:
                flag = False
            config["order_seed"] = 80
            l = list(range(config["n_class"]))
            random.seed(config["order_seed"])
            random.shuffle(l)
            setup_seed(config["seed"])
            config[
                "center_path"
            ] = f"./centerswithoutVar/CSQ_init_{flag}_{config['n_class']}_{hash_bit}.npy"

            config["save_path"] = f"./output/moco/{config['dataset']}/{hash_bit}"
            os.makedirs(config["save_path"], exist_ok=True)

            config[
                "save_center"
            ] = f"{config['save_path']}/ours_center.npy"

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{config["save_path"]}/train.log', rotation="500 MB", level="INFO")

            with open(f'{config["save_path"]}/config.json', 'w+') as f:
                json.dump(config, f, indent=4, sort_keys=True)

            best_epoch, best_map = train_val(config, hash_bit, l, train_loader, test_loader, database_loader, logger)

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
            f"[dataset:{x['dataset']}][hash-bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )
