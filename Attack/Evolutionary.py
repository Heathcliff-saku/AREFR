from cv2 import resize
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from config import Config as conf
from tqdm import tqdm
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import cv2 as cv 

def criterion(model, x, target, x_adv, flag_target):
    best_thresh = 0.3
    best_thresh_target = 0.6
    D = torch.norm(x-x_adv)
    if flag_target:
        f1 = model(target).squeeze()
        f2 = model(x_adv).squeeze()
        cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
        if cosdistance > best_thresh_target:
            C = 0
        else:
            C = float('inf')
    else:
        f1 = model(x).squeeze()
        f2 = model(x_adv).squeeze()
        cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
        if cosdistance < best_thresh:
            C = 0
            # print(cosdistance)
        else:
            C = float('inf')

    return D+C


class Evolutionary(nn.Module):
    """  Decision black box attack based on covariance 
         matrix adaptation evolution strategy ((1+1)-CMA-ES) for AREFR: -- v 1.0
    
    1、It's a decision based black box attack Use fewer access times to implement attacks
    2、Needn't to know the structure and parameter information of the network
    3、design an appropriate distribution to sample the random noise in each iteration

    Support:
        a、goal: 'Dod'(dodging) or 'Imp'(Impersonation)

    Reference: a、Dong Y, Su H, Wu B, et al. Efficient Decision-based Black-box Adversarial Attacks on Face Recognition[J]. IEEE, 2019.
               b、https://github.com/thu-ml/ares.git
               c、https://github.com/sgmath12/efficient-decision-based-black-box-adversarial-attacks-on-face-recognition.git

    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, interate=1000, flag_target=True, m=64, c_c=0.01, c_cov=0.001, sigma=3e-2, channels=1):
        """
        :param model:       The victim model(Only support nn.Model)
        :param interate:    The number of iterations
        :param flag_target: Whether there is a targeted attack
        :param m:           The lower dimension of the search space 
        :param c_c:         hyper-parameters of CMA
        :param c_cov:       hyper-parameters of CMA
        :param sigma:       Parameters used to construct a multidimensional Gaussian distribution
        """
        super(Evolutionary, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.interate = interate
        self.m =channels * m * m
        self.k = self.m // 20
        self.c_c = c_c
        self.c_cov = c_cov
        self.sigma = sigma
        self.flag_target = flag_target

    def generate(self, x, target):
        """
        :param x:      Clean sample's image
        :param target: The img of the attack target people (Same as above)
        :return:       adv_x
        """
        # Mu is a key hyperparameter that needs to be adjusted dynamically within the loop
        mu = 0.01
        p_c = torch.zeros(self.m)
        C = torch.eye(self.m)
        H, W = 128, 128
        success_rate = 0

        if self.flag_target:
            x_adv = target
        else:
            # x_adv = torch.rand_like(x)
            x_adv = x + torch.Tensor(np.random.uniform(-0.5, 0.5, x.shape)).type_as(x)

        x_adv.requires_grad = True

        for t in tqdm(range(self.interate)):
            z = MultivariateNormal(loc=torch.zeros([self.m]), covariance_matrix=(self.sigma ** 2)*C).rsample()
            zeroIdx = np.argsort(-C.diagonal())[self.k:]
            z[zeroIdx] = 0
            
            z = z.reshape([1, 1, 64, 64])
            # upsampling
            z_ = F.interpolate(z, (H,W), mode='bilinear', align_corners=True)
            z_ = z_ + mu * (x-x_adv)
            L_after = criterion(self.model, x, target, x_adv+z_, flag_target=self.flag_target)
            L_before = criterion(self.model, x, target, x_adv, flag_target=self.flag_target)

            if L_after < L_before:
                # update x_adv、C，
                x_adv = x_adv + z_
                p_c = (1-self.c_c) * p_c + np.sqrt(2*(2-self.c_c)) * z.reshape(-1)/self.sigma
                C[range(self.m), range(self.m)] = (1 - self.c_cov) * C.diagonal() + self.c_cov * (p_c) ** 2
                print (L_after)
                success_rate += 1
            
            if t % 30 == 0:
                mu = mu * np.exp(success_rate/30 - 1/5)
                success_rate = 0
            
            if t % 500 == 0:
                img = x_adv.cpu().detach().numpy()[0, 0, :, :]
                img = (img*0.5+0.5) * 255
                cv.imwrite('./fig/'+'iter_'+str(t)+'.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 100])
                

        pertubation = (x_adv - x).cpu().detach().numpy()
        x_adv = x_adv.cpu().detach().numpy()
        
        return x_adv, pertubation


