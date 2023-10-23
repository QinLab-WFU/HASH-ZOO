import json
import os

import torch.multiprocessing
from loguru import logger
from torchvision import transforms

from DPSH import DPSH_algo
from _data import build_loader


def build_loader_local(param):

    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_loader, test_loader, database_loader, topk, num_classes = build_loader(param["dataset_root"],
                                                                                 param["dataset"],
                                                                                 param["batch_size"], 4,
                                                                                 transformations,
                                                                                 transformations)

    param["topk"] = topk
    param["class_num"] = num_classes

    return train_loader, test_loader, database_loader


def get_config():
    param = {}
    param["lambda"] = 10
    param["dataset_root"] = "/home/sxz/Downloads/datasets"
    param["batch_size"] = 128
    param["epochs"] = 150
    param["learning_rate"] = 0.05
    param["weight_decay"] = 1e-5  # 10 ** -5
    param['model'] = 'alexnet'
    return param


if __name__ == "__main__":

    param = get_config()

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        logger.info(f"processing dataset: {dataset}")
        param["dataset"] = dataset

        train_loader, test_loader, database_loader = build_loader_local(param)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f"processing hash-bit: {hash_bit}")
            param["bit"] = hash_bit

            param["save_dir"] = f"./output/{param['model']}/{dataset}/{hash_bit}"
            os.makedirs(param["save_dir"], exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(
                f"{param['save_dir']}/train.log", rotation="500 MB", level="INFO"
            )

            # logger.info(config)
            with open(f'{param["save_dir"]}/config.json', 'w+') as f:
                json.dump(param, f, indent=4, sort_keys=True)

            best_epoch, best_map = DPSH_algo(param, train_loader, test_loader, database_loader, logger, 1)
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
