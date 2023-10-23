from copy import deepcopy

import torch
from torch import optim
from tqdm.auto import tqdm

from CenterHashing.loss.ourLoss import OurLoss
from CenterHashing.model.Net import MoCo
from _utils import prediction, mean_average_precision


def train_val(config, bit, l, train_loader, test_loader, database_loader, logger):

    net = MoCo(config, bit, config["label_size"]).cuda()
    if config["n_gpu"] > 1:
        net = torch.nn.DataParallel(net)
    optimizer = optim.RMSprop(
        net.parameters(), **(config["optimizer"]["optim_param"])
    )
    config["num_train"] = len(train_loader.dataset)

    criterion = OurLoss(config, bit, l)
    print(f"dmin: {criterion.d_min}, dmax: {criterion.d_max}")

    best_map = 0
    best_epoch = 0
    best_checkpoint = None
    count = 0

    for epoch in range(config["epoch"]):
        net.train()

        train_loss = 0
        train_center_loss = 0
        train_pair_loss = 0
        for img, label, ind in tqdm(train_loader):
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()

            u1, u2 = net(img, None, None)
            loss, center_loss, pair_loss = criterion(u1, u2, label, ind, epoch)
            if config["n_gpu"] > 1:
                loss = loss.mean()
                center_loss = center_loss.mean()
                if type(pair_loss) != int:
                    pair_loss = pair_loss.mean()
            train_loss += loss.item()
            train_center_loss += center_loss.item()
            if epoch >= config["epoch_change"]:
                train_pair_loss += pair_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config["max_norm"])
            optimizer.step()

        train_loss /= len(train_loader)
        logger.info(
            f"[Train][dataset:{config['dataset']}][bits:{bit}][epoch:{epoch}/{config['epoch'] - 1}][train_loss:{train_loss}][center_loss:{train_center_loss}][pair_loss: {train_pair_loss}]")

        if (epoch + 1) % config["test_map"] == 0 or (epoch + 1) == config['epoch']:
            qB, qL = prediction(net, test_loader, 0)
            rB, rL = prediction(net, database_loader, 0)
            map = mean_average_precision(qB, rB, qL, rL, config["topk"])
            logger.info(
                f"[Evaluation][dataset:{config['dataset']}][bits:{bit}][epoch:{epoch}/{config['epoch'] - 1}][best-mAP@{config['topk']}:{best_map:.7f}][mAP@{config['topk']}:{map:.7f}][count:{0 if map > best_map else (count + 1)}]")

            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_checkpoint = deepcopy(net.state_dict())
                count = 0
            else:
                count += 1
                if count == config["stop_iter"]:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
                    torch.save(best_checkpoint, f"{config['save_path']}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{config['save_path']}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map
