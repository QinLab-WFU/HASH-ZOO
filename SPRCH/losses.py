import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, loss=1, temperature=0.3, data_class=10):
        super(SupConLoss, self).__init__()
        self.loss = loss
        self.temperature = temperature
        self.data_class = data_class

    def forward(self, features, prototypes, labels=None, epoch=0, opt=None):
        # data-to-data
        anchor_feature = features
        contrast_feature = features
        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask
        neg_mask = 1 - mask
        anchor_dot_contrast = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T)
        all_exp = torch.exp(anchor_dot_contrast / self.temperature)
        pos_exp = pos_mask * all_exp
        neg_exp = neg_mask * all_exp
        # data-to-class
        pos_mask2 = labels
        neg_mask2 = 1 - labels
        anchor_dot_prototypes = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(prototypes, dim=1).T)
        all_exp2 = torch.exp(anchor_dot_prototypes / self.temperature)
        pos_exp2 = pos_mask2 * all_exp2
        neg_exp2 = neg_mask2 * all_exp2

        if opt.self_paced:
            if epoch <= int(opt.epochs/3):
                delta = epoch / int(opt.epochs/3)
            else:
                delta = 1
            pos_exp *= torch.exp(-1 - anchor_dot_contrast).detach() ** (delta/4)
            neg_exp *= torch.exp(-1 + anchor_dot_contrast).detach() ** (delta)
            pos_exp2 *= torch.exp(-1 - anchor_dot_prototypes).detach() ** (delta/4)
            neg_exp2 *= torch.exp(-1 + anchor_dot_prototypes).detach() ** (delta)

        if self.loss == 'p2p':
            loss = -torch.log(pos_exp.sum(1)/(neg_exp.sum(1) + pos_exp.sum(1)))
            return loss.mean()
        if self.loss == 'p2c':
            loss = -torch.log(pos_exp2.sum(1)/(neg_exp2.sum(1) + pos_exp2.sum(1)))
            return loss.mean()
        if self.loss == 'RCH':
            # balance two kinds of pairs
            if opt.weighting:
                lambda_pos = pos_mask.sum(1)/pos_mask2.sum(1)
                lambda_neg = neg_mask.sum(1)/neg_mask2.sum(1)
                loss = -torch.log((pos_exp.sum(1) + lambda_pos*pos_exp2.sum(1))
                                  / (neg_exp.sum(1) + lambda_neg*neg_exp2.sum(1) + pos_exp.sum(1) + lambda_pos*pos_exp2.sum(1)))
            else:
                loss = -torch.log((pos_exp.sum(1) + pos_exp2.sum(1))
                                  / (neg_exp.sum(1) + neg_exp2.sum(1) + pos_exp.sum(1) + pos_exp2.sum(1)))
            return loss.mean()









