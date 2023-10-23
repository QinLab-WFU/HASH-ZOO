import numpy as np
import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=10):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target):
        target_smooth = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)


class My_Loss(nn.Module):
    def __init__(self, num_classes, hash_code_length, mixup_fn, smoothing, alph, beta, gamm):
        super().__init__()
        self.hash_code_length = hash_code_length
        self.alph = alph
        self.beta = beta
        self.gamm = gamm

        if mixup_fn is not None:
            self.classify_loss_fun = SoftTargetCrossEntropy()
        elif smoothing > 0.:
            self.classify_loss_fun = LabelSmoothingLoss(epsilon=smoothing, num_classes=num_classes)
        else:
            self.classify_loss_fun = torch.nn.CrossEntropyLoss()

    def hash_loss(self, hash_out, target):
        theta = torch.einsum('ij,jk->ik', hash_out, hash_out.t()) / 2
        # one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        # one_hot = one_hot.float()
        Sim = (torch.einsum('ij,jk->ik', target, target.t()) > 0).float()

        pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)

        mask_positive = Sim > 0
        mask_negative = Sim <= 0
        S1 = mask_positive.float().sum() - hash_out.shape[0]
        S0 = mask_negative.float().sum()
        if S0 == 0:
            S0 = 1
        if S1 == 0:
            S1 = 1
        S = S0 + S1
        pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
        pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

        diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
        pair_loss = pair_loss - diag_matrix
        count = (hash_out.shape[0] * (hash_out.shape[0] - 1) / 2)

        return pair_loss.sum() / 2 / count

    def quanti_loss(self, hash_out):
        regular_term = (hash_out - hash_out.sign()).pow(2).mean()
        return regular_term

    def forward(self, hash_out, cls_out, target):
        cls_loss = self.classify_loss_fun(cls_out, target)
        hash_loss = self.hash_loss(hash_out, target)
        quanti_loss = self.quanti_loss(hash_out)
        loss = self.gamm * cls_loss + self.alph * hash_loss
        return hash_loss, quanti_loss, cls_loss, loss
