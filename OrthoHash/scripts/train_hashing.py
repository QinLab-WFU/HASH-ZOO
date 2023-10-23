from collections import defaultdict
from copy import deepcopy

import torch
from timm.utils import AverageMeter

from OrthoHash import configs
from OrthoHash.functions.hashing import get_hamm_dist
from OrthoHash.functions.loss.orthohash import OrthoHashLoss
from _utils import prediction, mean_average_precision


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)


def get_codebook(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.01):
    """
    brute force to find centroid with furthest distance
    :param nclass:
    :param nbit:
    :param maxtries:
    :param initdist:
    :param mindist:
    :param reducedist:
    :return:
    """
    codebook = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    while i < nclass:
        print(i, end='\r')
        c = torch.randn(nbit).sign()
        nobreak = True
        for j in range(i):
            if get_hd(c, codebook[j]) < currdist:
                i -= 1
                nobreak = False
                break
        if nobreak:
            codebook[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            print('reduce', currdist, i)
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1
    codebook = codebook[torch.randperm(nclass)]
    return codebook


def calculate_accuracy(logits, hamm_dist, labels, loss_param):
    if loss_param['multiclass']:
        pred = logits.topk(5, 1, True, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        acc = correct[:5].reshape(-1).float().sum(0, keepdim=True) / logits.size(0)

        pred = hamm_dist.topk(5, 1, False, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        cbacc = correct[:5].reshape(-1).float().sum(0, keepdim=True) / hamm_dist.size(0)
    else:
        acc = (logits.argmax(1) == labels.argmax(1)).float().mean()
        cbacc = (hamm_dist.argmin(1) == labels.argmax(1)).float().mean()

    return acc, cbacc


def train_hashing(optimizer, model, codebook, train_loader, loss_param):
    model.train()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)

    criterion = OrthoHashLoss(**loss_param)

    for i, (data, labels, _) in enumerate(train_loader):
        # clear gradient
        optimizer.zero_grad()

        data, labels = data.to(device), labels.to(device)
        logits, codes = model(data)

        loss = criterion(logits, codes, labels)

        # backward and update
        loss.backward()
        optimizer.step()

        hamm_dist = get_hamm_dist(codes, codebook, normalize=True)
        acc, cbacc = calculate_accuracy(logits, hamm_dist, labels, loss_param)

        # store results
        meters['loss_total'].update(loss.item(), data.size(0))
        meters['loss_ce'].update(criterion.losses['ce'].item(), data.size(0))
        meters['loss_quan'].update(criterion.losses['quan'].item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['cbacc'].update(cbacc.item(), data.size(0))

    return meters

def prepare_model(config, device, codebook=None):
    # logger.info('Creating Model')
    model = configs.arch(config, codebook=codebook)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model


def main(config, train_loader, test_loader, db_loader, logger):
    device = torch.device(config.get('device', 'cuda:0'))

    nclass = config['arch_kwargs']['nclass']
    nbit = config['arch_kwargs']['nbit']

    if config['codebook_generation'] == 'N':  # normal
        codebook = torch.randn(nclass, nbit)
    elif config['codebook_generation'] == 'B':  # bernoulli
        prob = torch.ones(nclass, nbit) * 0.5
        codebook = torch.bernoulli(prob) * 2. - 1.
    else:  # O: optim
        codebook = get_codebook(nclass, nbit)

    codebook = codebook.sign().to(device)
    torch.save(codebook, f'{config["save_dir"]}/codebook.pth')

    # train_loader, test_loader, db_loader = prepare_dataloader(config)
    model = prepare_model(config, device, codebook)
    # print(model)

    backbone_lr_scale = 0.1
    optimizer = configs.optimizer(config, [{'params': model.get_backbone_params(),
                                            'lr': config['optim_kwargs']['lr'] * backbone_lr_scale},
                                           {'params': model.get_hash_params()}])
    scheduler = configs.scheduler(config, optimizer)

    loss_param = config.copy()
    loss_param.update({'device': device})

    best_map = 0
    best_epoch = 0
    best_checkpoint = None
    count = 0

    for epoch in range(config['epochs']):

        train_meters = train_hashing(optimizer, model, codebook, train_loader, loss_param)
        scheduler.step()

        out_str = f"[Train][dataset:{config['dataset']}][bits:{nbit}][epoch:{epoch}/{config['epochs'] - 1}]"
        for key in train_meters:
            out_str += f"[{key}:{train_meters[key].avg}]"
        logger.info(out_str)

        if (epoch + 1) % config['eval_interval'] == 0 or (epoch + 1) == config['epochs']:
            model.eval()
            qB, qL = prediction(model, test_loader, 1)
            rB, rL = prediction(model, db_loader, 1)
            map = mean_average_precision(qB, rB, qL, rL, config['R'])
            logger.info(
                f"[Evaluation][dataset:{config['dataset']}][bits:{nbit}][epoch:{epoch}/{config['epochs'] - 1}][best-mAP@{config['R']}:{best_map:.7f}][mAP@{config['R']}:{map:.7f}][count:{0 if map > best_map else (count + 1)}]")

            if map > best_map:
                best_map = map
                best_epoch = epoch
                best_checkpoint = deepcopy(model.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
                    torch.save(best_checkpoint, f"{config['save_dir']}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{config['save_dir']}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map
