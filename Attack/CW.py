from cmath import inf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.loss import FocalLoss
from model.metric import ArcFace, CosFace
from config import Config as conf
from .Loss import FaceCWLoss
from tqdm import tqdm

def arctanh(x, eps=1e-6):
    # Inverse hyperbolic tangent function
    """
    tanh(x) = e^x-e^(-x) /  e^x+e^(-x)
    arctanh(x) = 0.5 * lg((x+1) / (-x+1))
    """
    x = x - eps
    return 0.5 * torch.log((x+1.0) / (-x+1.0))

def tanh2x(x, box_max, box_min):
    # Convert the variable x from the tanh space into the number field
    box_mul = (box_max - box_min) / 2
    box_plus = (box_max + box_min) / 2
    return arctanh((x-box_plus) / box_mul)


def x2tanh(x, box_max, box_min):
    # Convert the variable x from the number field into the tanh space
    box_mul = (box_max - box_min) / 2
    box_plus = (box_max + box_min) / 2
    return torch.tanh(x) * box_mul + box_plus


class FaceCW(nn.Module):
    """ Carlini & Wagner Attack (C&W) for AREFR: -- v 1.0
    1、It is a adversarial sample generation method based on optimization
    2、The disturbance is smoother and is able to strike the defensive distillation model
    3、The sample is mapped to tanh space for optimization
    
    Support:
        a、goal: 'target' or 'no_target'
        b、distance metric: 'l_2' or 'l_inf'
        c、optimizer: 'sgd' or 'adam'
    
    Reference: a、Carlini N, Wagner D. Towards Evaluating the Robustness of Neural Networks[J]. 2017 IEEE Symposium on Security and Privacy (SP), 2017.
               b、ares: https://github.com/thu-ml/ares.git
    
    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, lr=1e-2, interate=100, binary_search_steps=10, c_range=(1e-3, 1e10),
                    confidence=0.0, abort_early=True, random_star=True, distance='L2',
                    optimizer='Adam', flag_target=True):
        """
        :param model:       The victim model(Only support nn.Model)
        :param lr:          Optimized iteration step size (learning rate)
        :param interate:    The number of iterations
        :param binary_search_steps: Find the number of searches for the tradeoff parameter c between losses
        :param c_range:     The search range for parameter c
        :param confidence:  Robustness of adversarial samples
        :param abort_early: if stop improving, abort gradient descent early
        :param random_star: Whether to add random noise at the beginning
        :param distance:    Distance metric norm
        :param optimizer:   Optimizer
        :param flag_target: Whether there is a targeted attack
        """
        super(FaceCW, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.lr = lr
        self.interate = interate
        self.binary_search_steps = binary_search_steps
        self.K = confidence
        self.abort_early = abort_early
        self.random_star = random_star
        self.c_range = c_range
        self.optimizer = optimizer

        self.flag_target = flag_target
        # Box constraint, the pixel value of the image is 0~1
        """
        Why -1 and 1: because pixel space is shifted from [0,1] to [-1,1]
        during image preprocessing use:
        x = (x-mean) / std
        """
        self.box_min = -1.0
        self.box_max = 1.0

        self.loss_func = FaceCWLoss()
        self.attack_mertic = distance

        self.eps = 0.01
    
    def generate(self, x, target):
        """
        :param x:      Clean sample's image
        :param target: The img of the attack target people (Same as above)
        :return:       adv_x

        The Loss Function for L2-FaceCW Attack:
            loss = |delta|^2_2 + c·L_cw(x+delta, t)
            where: delta = tanh(x) * box_mul + box_plus
                   L_cw = max(max(cos<x_adv', t'>, cos<x_adv', x'>)-cos<x_adv', t'>, -k)               
                   cos<·> is cosin distance
                   x_adv'，x'， t' is embeddings of x_adv, x, t
        """

        # Initializing disturbance
        if self.random_star:
            pertubation = torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        else:
            pertubation = torch.zeros(x.shape).type_as(x).cuda()
        
        pertubation_tanh = x2tanh(pertubation, self.box_max, self.box_min)
        pertubation_tanh.requires_grad = True
        # pertubation_tanh.cuda()


        if self.optimizer == 'Adam':
            optimizer = optim.Adam([pertubation_tanh], lr=self.lr, weight_decay=conf.weight_decay)
        else:
            optimizer = optim.SGD([pertubation_tanh], lr=self.lr, weight_decay=conf.weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

        # Calculate clean sample and target embeddings  
        x_emb = self.model(x).squeeze()
        t_emb = self.model(target).squeeze()

        losss = []
        loss = inf
        # Parameters required for binary search
        lower_bounds = self.c_range[0]
        upper_bounds = self.c_range[1]
        for i_outer in range(1):
            """
            The purpose of the outer loop is to find the most suitable parameter C
            """
            c = 0.5
            
            for i_inner in range(self.interate):
                optimizer.zero_grad()
                x_adv = x + tanh2x(pertubation_tanh.cpu(), self.box_max, self.box_min)
                x_adv_emb = self.model(x_adv).squeeze()

                # update loss func:
                if self.flag_target:
                    loss = self.loss_func(pertubation_tanh, x_adv_emb, x_emb, t_emb, c, self.K)
                    loss = loss.cuda()
                    losss.append(loss.cpu().detach().numpy())
                else:
                    loss = self.loss_func(pertubation_tanh, x_adv_emb, x_emb, t_emb, c, self.K)
                    loss = loss.cuda()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
        
        x_adv = x + tanh2x(pertubation_tanh.cpu(), self.box_max, self.box_min)
        x_adv = x_adv.detach().numpy()
        pertubation = tanh2x(pertubation_tanh.cpu(), self.box_max, self.box_min)
        pertubation = pertubation.detach().numpy()
        # print(losss)
        return x_adv, pertubation