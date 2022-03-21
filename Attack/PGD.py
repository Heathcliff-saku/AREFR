import torch
import torch.nn as nn
import numpy as np
from model.loss import FocalLoss
from model.metric import ArcFace, CosFace
from config import Config as conf
from .Loss import CosLoss
from tqdm import tqdm

class FacePGD(nn.Module):
    """  Projected Gradient Descent (PGD) for AREFR: -- v 1.0
    1、 The strongest first-order attack algorithm
    2、 Compared with BIM, PGD has no limit on the step size
    and can iterate freely for any number of times. Each time, it will project back to the constraint space
    3、 There are steps to initialize random perturbations

    Support:
        a、goal: 'target' or 'no_target'
        b、distance metric: 'l_2' or 'l_inf'
        b、random perturbations:'True' or 'False'

    Reference: a、Madry A, Makelov A, Schmidt L, et al. Towards Deep Learning Models Resistant to Adversarial Attacks[J]. 2017.
               b、ares: https://github.com/thu-ml/ares.git

    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, eps=0.03, a=0.001, interate=40, random_star=True, distance=False, clip_min=-1.0, clip_max=1.0, flag_target=True):
        """
        :param model:       The victim model(Only support nn.Model)
        :param eps:         The upper bound of the disturbance
        :param a:           Disturbance coefficient of each iteration
        :param interate:    The number of iterations
        :param random_star: Whether to add random noise at the beginning
        :param distance:    Distance metric norm
        :param clip_min:    Lower bound of clip function
        :param clip_max:    upper bound of clip function
        :param flag_target: Whether there is a targeted attack
        """
        super(FacePGD, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.eps = eps
        self.a = a
        self.interate = interate
        self.random_star = random_star
        self.loss_func = CosLoss()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.flag_target = flag_target

    def generate(self, x, target):
        """
        :param x:      Clean sample's image
        :param target: The img of the attack target people (Same as above)
        :return:       adv_x
        """
        if self.random_star:
            x_new = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x)
            x_new.requires_grad = True
        else:
            x_new = x

        # pertubation = torch.zeros(x.shape).type_as(x)
        embedding_true = self.model(x).squeeze()
        target_embedding = self.model(target).squeeze()

        for i in tqdm(range(self.interate)):
            # x_adv = x_new + pertubation
            embedding_new = self.model(x_new)
            embedding_new = embedding_new.squeeze()

            if self.flag_target:
                loss = -self.loss_func(target_embedding, embedding_new)
            else:
                loss = self.loss_func(embedding_true, embedding_new)

            # print("Loss:", loss)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            # x_new = x_new.squeeze()

            grad = x_new.grad.cpu().detach().numpy()
            grad = np.sign(grad)
            pertubation = grad * self.a

            x_new = x_new.cpu().detach().numpy() + pertubation
            x = x.cpu().detach().numpy()

            # clip
            x_new = np.clip(x_new, x - self.eps, x + self.eps)
            x_new = np.clip(x_new, self.clip_min, self.clip_max)

            # pertubation after clip
            pertubation = x_new - x

            x = torch.Tensor(x)
            x_new = torch.Tensor(x_new).type_as(x).cuda()
            x_new.requires_grad = True

        x_adv = x_new.cpu().detach().numpy()
        pertubation = pertubation
        return x_adv, pertubation

