#coding=UTF-8

import torch
import torch.nn as nn

class Cyclical_FocalLoss(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, gamma_hc=0, eps: float = 0.1, reduction='mean', epochs=200, 
                 factor=2):
        super(Cyclical_FocalLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_hc = gamma_hc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.epochs = epochs
        self.factor = factor # factor=2 for cyclical, 1 for modified
#        self.ceps = ceps
#        print("Asymetric_Cyclical_FocalLoss: gamma_pos=", gamma_pos," gamma_neg=",gamma_neg, 
#              " eps=",eps, " epochs=", epochs, " factor=",factor)

    def forward(self, inputs, target, epoch):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
#        print("input.size(),target.size()) ",inputs.size(),target.size())
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        if len(list(target.size()))>1:
            target = torch.argmax(target, 1)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # Cyclical
#        eta = abs(1 - self.factor*epoch/(self.epochs-1))
        if self.factor*epoch < self.epochs:
            eta = 1 - self.factor *epoch/(self.epochs-1)
        elif self.factor == 1.0:
            eta = 0
        else:
            eta = (self.factor*epoch/(self.epochs-1) - 1.0)/(self.factor - 1.0) 

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        positive_w = torch.pow(1 + xs_pos,self.gamma_hc * targets)
        log_preds = log_preds * ((1 - eta)* asymmetric_w + eta * positive_w)

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss