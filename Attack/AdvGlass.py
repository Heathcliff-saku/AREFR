import torch
import torch.nn as nn
import numpy as np
from model.loss import FocalLoss
from model.metric import ArcFace, CosFace
from config import Config as conf
from .Loss import AdvGlassLoss
from tqdm import tqdm


class AdvGlass(nn.Module):
    """ Adversarial Eyeglasses Attack for AREFR: -- v 1.0
    1、Is a physical environment attacks for the face recognition system
    2、used PGD to generate robust anti-eyewear patch in physical environment to realize attack

    Support:
        a、goal: 'target' or 'no_target'
        b、random perturbations:'True' or 'False'

    Reference: a、Sharif M, Bhagavatula S, Bauer L, et al. Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition. ACM SIGSAC CCS '16，2016, 1528–1540.
               b、ares: https://github.com/thu-ml/ares.git

    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, eps=0.03, a=0.01, interate=400, random_star=True, distance=False, clip_min=-1.0, clip_max=1.0, flag_target=True):
        """
        :param model:       The victim model(Only support nn.Model)
        :param eps:         The upper bound of the disturbance
        :param a:           step size of each iteration
        :param interate:    The number of iterations
        :param random_star: Whether to add random noise at the beginning
        :param distance:    Distance metric norm
        :param clip_min:    Lower bound of clip function
        :param clip_max:    upper bound of clip function
        :param flag_target: Whether there is a targeted attack
        """
        super(AdvGlass, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.eps = eps
        self.a = a
        self.interate = interate
        self.random_star = random_star
        self.loss_func = AdvGlassLoss()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.flag_target = flag_target

    def generate(self, x_batch, target):
        """
        :param x_batch: Series of Clean sample's image of one person  
        :param target:  The img of the attack target people (Same as above)
        :return:        adv_x
        """
        if self.random_star:
            x_new = x_batch + torch.Tensor(np.random.uniform(-self.eps, self.eps, x_batch.shape)).type_as(x_batch)
            x_new.requires_grad = True
        else:
            x_new = x_batch

        pertubation = x_new[0, :, :, :] - x_batch[0, :, :, :]

        embedding_true = self.model(x_batch).squeeze()
        target_embedding = self.model(target).squeeze()

        for i in range(self.interate):
            # x_adv = x_new + pertubation
            embedding_new = self.model(x_new)
            embedding_new = embedding_new.squeeze()

            if self.flag_target:
                loss = - self.loss_func(target_embedding, embedding_new, pertubation)
            else:
                loss = self.loss_func(embedding_true, embedding_new, pertubation)
            print('total loss: ', loss)

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            # x_new = x_new.squeeze()

            grad = x_new.grad.cpu().detach().numpy()
            grad = np.sign(grad)
            pertubation = grad * self.a

            x_new = x_new.cpu().detach().numpy() + pertubation
            x_batch = x_batch.cpu().detach().numpy()

            # clip
            x_new = np.clip(x_new, x_batch - self.eps, x_batch + self.eps)
            x_new = np.clip(x_new, self.clip_min, self.clip_max)

            # pertubation after clip
            pertubation = x_new[0, :, :, :] - x_batch[0, :, :, :]
            pertubation = torch.from_numpy(pertubation).cuda()
            pertubation.requires_grad = True

            x_batch = torch.Tensor(x_batch)
            x_new = torch.Tensor(x_new).type_as(x_batch).cuda()
            x_new.requires_grad = True

        x_adv = x_new.cpu().detach().numpy()
        pertubation = pertubation.cpu().detach().numpy()
        return x_adv, pertubation