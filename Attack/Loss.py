import torch
import torch.nn as nn
import numpy as np


class CosLoss(nn.Module):
    """
    Cosloss is used for FGSM/PGD
    It measures the similarity between facial features by cos-distance
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = 1 - (torch.dot(input, target) / (torch.norm(input, p=2) * (torch.norm(target, p=2))))
        return torch.mean(loss).requires_grad_(True)

def cosin(x1, x2):
    return torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))

class FaceCWLoss(nn.Module):
    """
    FaceCWLoss is used for CW Attack
    It contains two parts: Error loss and similarity loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pertubation, x_adv_emb, x_emb, t_emb, c, k):
        loss1 = torch.norm(pertubation)
        adv_t = cosin(x_adv_emb, t_emb)
        adv_x = cosin(x_adv_emb, x_emb)
        # loss2 = max(torch.max(adv_t, adv_x) - adv_t, -k)
        loss2 = 3*(1-adv_t) + adv_x
        return loss1 + c * loss2


class AdvGlassLoss(nn.Module):
    """
    AdvGlassLoss is used for galss attack
    It contains three parts: CosLoss、TVLoss、NPSLoss
    """
    def __init__(self):
        super().__init__()

    def _tensor_size(self, t):
        return t.size()[0] * t.size()[1] * t.size()[2]

    def forward(self, target, input_batch, pertubation):
        Cosloss = 0
        for i in range(input_batch.shape[0]):
            input = input_batch[i, :].squeeze()    
            Cosloss += torch.mean(1 - (torch.dot(input, target) / (torch.norm(input, p=2) * (torch.norm(target, p=2)))))
        Cosloss = Cosloss / input_batch.shape[0]

        h_x = pertubation.size()[1]
        w_x = pertubation.size()[2]
        count_h = self._tensor_size(pertubation[:, 1:, :]) 
        count_w = self._tensor_size(pertubation[:, :, 1:])
        h_tv = torch.pow((pertubation[:, 1:, :] - pertubation[:, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((pertubation[:, :, 1:] - pertubation[:, :, :w_x - 1]), 2).sum()
        TVloss = 2 * (h_tv / count_h + w_tv / count_w)
        
        print('Cosloss: ',  Cosloss)
        print('TVloss: ', TVloss)

        return Cosloss.requires_grad_(True) + 100 * TVloss.requires_grad_(True)