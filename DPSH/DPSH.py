import os
from copy import deepcopy

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

import CNN_model
from _utils import prediction, mean_average_precision


def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S


def CreateModel(model_name, bit, use_gpu):
    if model_name == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bit)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bit)
    if use_gpu:
        cnn_model = cnn_model.cuda()
    return cnn_model


def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt


def Totloss(U, B, Sim, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta * theta).sum() / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(Variable(theta), False).data).sum()
    l2 = (U - B).pow(2).sum()
    l = l1 + lamda * l2
    return l, l1, l2, t1


def DPSH_algo(param, train_loader, test_loader, database_loader, logger, gpu_ind=0):
    # parameters setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)

    learning_rate = param["learning_rate"]
    weight_decay = param["weight_decay"]
    lamda = param['lambda']
    use_gpu = torch.cuda.is_available()

    ### create model
    model = CreateModel(param['model'], param['bit'], use_gpu)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    ### training phase
    # parameters setting
    num_train = len(train_loader.dataset)
    B = torch.zeros(num_train, param['bit'])
    U = torch.zeros(num_train, param['bit'])
    train_labels = train_loader.dataset.get_onehot_targets()

    totloss_record = []
    totl1_record = []
    totl2_record = []
    t1_record = []

    Sim = CalcSim(train_labels, train_labels)

    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(param['epochs']):
        model.train()
        epoch_loss = 0.0
        ## training epoch
        for train_input, train_label, batch_ind in train_loader:
            if use_gpu:
                S = CalcSim(train_label, train_labels)
                train_input = Variable(train_input.cuda())
            else:
                S = CalcSim(train_label, train_labels)
                train_input = Variable(train_input)

            model.zero_grad()
            train_outputs = model(train_input)
            for i, ind in enumerate(batch_ind):
                U[ind, :] = train_outputs.data[i]
                B[ind, :] = torch.sign(train_outputs.data[i])

            Bbatch = torch.sign(train_outputs)
            if use_gpu:
                theta_x = train_outputs.mm(Variable(U.cuda()).t()) / 2
                logloss = (Variable(S.cuda()) * theta_x - Logtrick(theta_x, use_gpu)).sum() / (
                        num_train * len(train_label))
                regterm = (Bbatch - train_outputs).pow(2).sum() / (num_train * len(train_label))
            else:
                theta_x = train_outputs.mm(Variable(U).t()) / 2
                logloss = (Variable(S) * theta_x - Logtrick(theta_x, use_gpu)).sum() / (num_train * len(train_label))
                regterm = (Bbatch - train_outputs).pow(2).sum() / (num_train * len(train_label))

            loss = - logloss + lamda * regterm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch, param['epochs'] - 1, epoch_loss / num_train))
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

        l, l1, l2, t1 = Totloss(U, B, Sim, lamda, num_train)
        totloss_record.append(l)
        totl1_record.append(l1)
        totl2_record.append(l2)
        t1_record.append(t1)

        logger.info('[Total Loss: %10.5f][total L1: %10.5f][total L2: %10.5f][norm theta: %3.5f]' % (l, l1, l2, t1))

        ## testing during epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == param['epochs']:
            qB, qL = prediction(model, test_loader)
            rB, rL = prediction(model, database_loader)
            map = mean_average_precision(qB, rB, qL, rL, param['topk'])
            logger.info(
                f"[Evaluation][dataset:{param['dataset']}][bits:{param['bit']}][epoch:{epoch}/{param['epochs'] - 1}][best-mAP@{param['topk']}:{best_map:.7f}][mAP@{param['topk']}:{map:.7f}][count:{0 if map > best_map else (count + 1)}]")

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
                    torch.save(best_checkpoint, f"{param['save_dir']}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{param['save_dir']}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map
