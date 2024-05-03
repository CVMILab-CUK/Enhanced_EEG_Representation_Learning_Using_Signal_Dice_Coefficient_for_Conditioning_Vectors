import torch
import torch.nn as nn
from libs.metric import SignalDice

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


class SignalDiceLoss(nn.Module):

    def __init__(self, sep=True,  eps=1e-6):
        self.sdsc = SignalDice(eps)
        self.sep = sep
    
    def forward(self, inputs, targets):
        sdsc_value = self.sdsc(inputs, targets)

        if self.sep:
            return torch.mean(1 - torch.mean((2*torch.sum(self.sdsc.intersection, dim=1) + self.eps) / (torch.sum(self.sdsc.union, dim=1) + self.eps), dim=1))
        else:
            return 1 - sdsc_value