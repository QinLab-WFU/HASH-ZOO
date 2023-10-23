import json
import os
from copy import deepcopy

import math
import torch.multiprocessing
import xlrd
from loguru import logger
from torch.backends import cudnn
from torchvision import transforms

from HyP2.config import get_config
from HyP2.model import HyP, AlexNet
from _data import build_loader
from _utils import prediction, mean_average_precision


def train(args, trainloader, test_loader, database_loader, device, logging):
    feature_model = AlexNet(hash_bit=args.hash_bit)
    feature_model.to(device)
    model = HyP(args.seed, args.num_classes, args.hash_bit, args.threshold, args.beta).to(device)
    optimizer = torch.optim.SGD([{'params': feature_model.parameters(), 'lr': args.feature_rate},
                                 {'params': model.parameters(), 'lr': args.rate}], momentum=0.9,
                                weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_map = 0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.epochs):
        feature_model.train()
        train_loss = 0
        for i, (images, labels, _) in enumerate(trainloader):
            batch_x = images.to(device)
            batch_y = labels.to(device)

            hash_value = feature_model(batch_x)
            loss = model(x=hash_value, batch_y=batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        train_loss /= len(train_loader)
        logging.info(
            f"[Train][dataset:{args.dataset}][bits:{args.hash_bit}][epoch:{epoch}/{args.epochs - 1}][train-loss:{train_loss}]")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            qB, qL = prediction(feature_model, test_loader)
            rB, rL = prediction(feature_model, database_loader)
            map = mean_average_precision(qB, rB, qL, rL, args.retrieve)
            logging.info(
                f"[Evaluation][dataset:{args.dataset}][bits:{args.hash_bit}][epoch:{epoch}/{args.epochs - 1}][best-mAP@{args.retrieve}:{best_map:.7f}][mAP@{args.retrieve}:{map:.7f}][count:{0 if map > best_map else (count + 1)}]")

            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_checkpoint = deepcopy(feature_model.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logging.info(
                        f"without improvement, will save & exit, best mAP@{args.retrieve}: {best_map}, best epoch: {best_epoch}")
                    torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logging.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def build_loader_local(args):
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_loader, test_loader, database_loader, topk, num_classes = build_loader("/home/sxz/Downloads/datasets",
                                                                                 args.dataset,
                                                                                 args.batch_size, 4,
                                                                                 data_transform['train'],
                                                                                 data_transform['val'])

    args.retrieve = topk
    args.num_classes = num_classes

    return train_loader, test_loader, database_loader


if __name__ == "__main__":

    args = get_config()

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    cudnn.benchmark = True

    # Device configuration
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')

    dummy_logger_id = None
    rst = []
    for dataset in ["nuswide", "flickr", "coco"]:
        logger.info(f'processing dataset: {dataset}')
        args.dataset = dataset

        train_loader, test_loader, database_loader = build_loader_local(args)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')
            args.hash_bit = hash_bit

            # find the value of ζ
            sheet = xlrd.open_workbook('./codetable.xls').sheet_by_index(0)
            args.threshold = sheet.row(hash_bit)[math.ceil(math.log(args.num_classes, 2))].value
            logger.info(f'threshold: {args.threshold}')

            # path for loading and saving models
            args.save_dir = f'./output/{args.backbone}/{dataset}/{hash_bit}'
            os.makedirs(args.save_dir, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{args.save_dir}/train.log', rotation="500 MB", level="INFO")

            with open(f'{args.save_dir}/config.json', 'w+') as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train(args, train_loader, test_loader, database_loader, device, logger)
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
