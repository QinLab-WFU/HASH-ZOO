import argparse
import json
import os
from copy import deepcopy

import torch
import torch.optim as optim
import torchvision
from loguru import logger

from SPRCH.losses import SupConLoss
from SPRCH.utils.evaluate import compute_ratio, feed_random_seed
from _data import build_loader
from _utils import prediction, mean_average_precision


def train(dataloader, net, optimizer, criterion, epoch, opt, emb):
    accum_loss = 0
    net.train()
    for _, (img, label, _) in enumerate(dataloader):
        features = net(img.cuda())
        features = torch.tanh(features)
        label = label.cuda()
        prototypes = emb(torch.eye(opt.data_class).cuda())
        prototypes = torch.tanh(prototypes)
        loss = criterion(features, prototypes, label, epoch, opt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accum_loss += loss.item()
    return accum_loss / len(dataloader)


def str2bool(str):
    return True if str.lower() == 'true' else False


def main(args, train_loader, test_loader, database_loader, logger):
    # setup net
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = torch.nn.Linear(512, args.binary_bits)
    net.cuda()

    class Embedding(torch.nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.Embedding = torch.nn.Linear(args.data_class, args.binary_bits)

        def forward(self, x):
            output = self.Embedding(x)
            return output

    emb = Embedding().cuda()

    # setup loss
    criterion = SupConLoss(loss=args.loss, temperature=args.temp, data_class=args.data_class).cuda()

    # setup optimizer
    hash_id = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in hash_id, net.parameters())
    optimizer = optim.Adam([{'params': feature_params, 'lr': args.lr},
                            {'params': emb.parameters(), 'lr': 100 * args.lr},
                            {'params': net.fc.parameters(), 'lr': 10 * args.lr}]
                           )

    # calculate pos_ratio and neg_ratio
    pos_ratio, neg_ratio = compute_ratio(train_loader)
    logger.info(f'mean ratio for postive pairs:{pos_ratio.mean():.2f}')
    logger.info(f'mean ratio for negative pairs:{neg_ratio.mean():.2f}')

    # training process
    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.epochs):
        train_loss = train(train_loader, net, optimizer, criterion, epoch, args, emb)
        logger.info(
            f"[Train][dataset:{args.data_name}][bits:{args.binary_bits}][epoch:{epoch}/{args.epochs - 1}][train_loss:{train_loss}]")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            # compute mAP by searching testset images from trainset
            net.eval()
            qB, qL = prediction(net, test_loader)
            rB, rL = prediction(net, database_loader)
            map_k = mean_average_precision(qB, rB, qL, rL, args.k)
            logger.info(
                f"[Evaluation][dataset:{args.data_name}][bits:{args.binary_bits}][epoch:{epoch}/{args.epochs - 1}][best-mAP@{args.k}:{best_map:.7f}][mAP@{args.k}:{map_k:.7f}][count:{0 if map_k > best_map else (count + 1)}]")

            if map_k > best_map:
                best_map = map_k
                best_epoch = epoch
                best_checkpoint = deepcopy(net.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
                    torch.save(best_checkpoint, f"{args.outf}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.outf}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def get_config():
    parser = argparse.ArgumentParser(description='SPRCH')

    parser.add_argument('--temp', type=float, default=0.3, help='temperature')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--loss', type=str, default='p2p', help='different kinds of loss')
    parser.add_argument('--weighting', type=str2bool, default='True', help='--balance two kinds of pairs')
    parser.add_argument('--self_paced', type=str2bool, default='True', help='--self_paced learning schedule')

    # these items is changed or will be changed
    parser.add_argument('--data_path', default='/home/sxz/Downloads/datasets', help='path to dataset')
    parser.add_argument('--data_name', type=str, default='coco', help='cifar or coco...')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--binary_bits', type=int, default=16, help='length of hashing binary')
    parser.add_argument('--data_class', type=int, default=10, help='the number of dataset classes')
    parser.add_argument('--outf', default='save', help='folder to output model checkpoints')
    parser.add_argument('--k', type=int, default=5000, help='mAP@k')

    return parser.parse_args()


def build_loader_local(args):
    train_loader, test_loader, database_loader, topk, num_classes = build_loader(args.data_path, args.data_name,
                                                                                 args.batch_size, args.num_workers)

    args.k = topk
    args.data_class = num_classes

    return train_loader, test_loader, database_loader


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = get_config()
    feed_random_seed()

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    dummy_logger_id = None
    rst = []
    for dataset in ['cifar', 'nuswide', 'flickr', 'coco']:
        logger.info(f'processing dataset: {dataset}')
        args.data_name = dataset

        train_loader, test_loader, database_loader = build_loader_local(args)

        for hash_bit in [16, 32, 48, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')
            args.binary_bits = hash_bit

            args.outf = f"./output/resnet18/{dataset}/{hash_bit}"
            os.makedirs(args.outf, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{args.outf}/train.log', rotation="500 MB", level="INFO", )

            with open(f'{args.outf}/config.json', 'w+') as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = main(args, train_loader, test_loader, database_loader, logger)
            rst.append(
                {
                    "dataset": dataset,
                    "hash_bit": hash_bit,
                    "best_epoch": best_epoch,
                    "best_map": best_map
                }
            )
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][hash-bit:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )
