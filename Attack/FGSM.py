from msilib.schema import BindImage
import torch
import torch.nn as nn
import numpy as np
from model.loss import FocalLoss
from model.metric import ArcFace, CosFace
from config import Config as conf
from .Loss import CosLoss
from tqdm import tqdm

"""
FGSM and its derivative algorithm: FGSM、BIM/I-FGSM、MI-FGSM
"""
def ToCpuNumpy(x):
    return x.cpu().detach().numpy()

class FaceFGSM(nn.Module):
    """ Fast Gradient Sign Method (FGSM) for AREFR -- v 1.0
    1、A successful attack algorithm based on gradient direction is proposed for the first time
    2、It only needs one iteration and is fast
    3、This proves goodfellow et al. 's linear explanation for the existence of adversarial samples 

    Reference:  a、Goodfellow I, Shlens J, Szegedy C. Explaining and Harnessing Adversarial Examples[J]. Computer Science, 2014.
                b、Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. 2016.
                c、ares: https://github.com/thu-ml/ares.git
    
    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, eps=0.03, random_star=True, clip_min=-1.0, clip_max=1.0, flag_target=True):
        """
        :param model:       The victim model(Only support nn.Model)
        :param eps:         The upper bound of the disturbance
        :param random_star: Whether to add random noise at the beginning
        :param clip_min:    Lower bound of clip function
        :param clip_max:    upper bound of clip function
        :param flag_target: Whether there is a targeted attack
        """
        super(FaceFGSM, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.eps = eps
        self.random_star = random_star
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.flag_target = flag_target
        self.loss_func = CosLoss()
      
    def generate(self, x, target):
        """
        :param x:      Clean sample's image
        :param target: The img of the attack target people (Same as above)
        :return:       adv_x
        """
        # Initializing pertubation
        if self.random_star:
            x = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x)
            x.requires_grad = True
        else:
            x.requires_grad = True

        x_emb = self.model(x).squeeze()
        t_emb = self.model(target).squeeze()

        if self.flag_target:
            loss = -self.loss_func(t_emb, x_emb)
        else:
            loss = 1-self.loss_func(x_emb, x_emb)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grad = ToCpuNumpy(x.grad)
        grad = np.sign(grad)

        pertubation = grad * self.eps
        x_adv = ToCpuNumpy(x) + pertubation

        # clip
        x_adv = np.clip(x_adv, self.clip_min, self.clip_max)
        return x_adv, pertubation


class FaceBIM(nn.Module):
    """ Basic Iterative Method (BIM) for AREFR -- v 1.0
    1、Iterative form of FGSM  
    2、The number of iterations is limited by perturbation size and step size
    3、There were no random initializers in the original version

    Support:
        a、goal: 'target' or 'no_target'
        b、random perturbations:'True' or 'False'

    Reference: a、Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. 2016.
               b、ares: https://github.com/thu-ml/ares.git
    
    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, eps=0.03, a=0.001, random_star=True, distance=False, clip_min=-1.0, clip_max=1.0, flag_target=True):
        """
        :param model:       The victim model(Only support nn.Model)
        :param eps:         The upper bound of the disturbance
        :param a:           step size of each iteration
        :param random_star: Whether to add random noise at the beginning
        :param distance:    Distance metric norm
        :param clip_min:    Lower bound of clip function
        :param clip_max:    upper bound of clip function
        :param flag_target: Whether there is a targeted attack
        """
        super(FaceBIM, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.eps = eps
        self.a = a
        # The number of iterations is limited by the eps and step size
        self.interate = int(self.eps/self.a)
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

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            # x_new = x_new.squeeze()

            grad = x_new.grad.cpu().detach().numpy()
            grad = np.sign(grad)
            pertubation = grad * self.a

            x_new = x_new.cpu().detach().numpy() + pertubation
            x = x.cpu().detach().numpy()

            # clip
            x_new = np.clip(x_new, self.clip_min, self.clip_max)

            # pertubation after clip
            pertubation = x_new - x

            x = torch.Tensor(x)
            x_new = torch.Tensor(x_new).type_as(x).cuda()
            x_new.requires_grad = True

        x_adv = x_new.cpu().detach().numpy()
        pertubation = pertubation
        return x_adv, pertubation



class FaceMIFGSM(nn.Module):
    """ Momentum iteration method (MIM/MI-FGSM) for AREFR -- v 1.0
    1、Introduce momentum into the iterative formula
    2、So that the loss changes more smoothly
    3、This method has good black box migration 

    Support:
        a、goal: 'target' or 'no_target'
        b、random perturbations:'True' or 'False'

    Reference: a、Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. 2016.
               b、ares: https://github.com/thu-ml/ares.git
    
    Author: Silvester_Ruan  1290481944@qq.com
            School of Artificial Intelligence, xidian university
    """
    def __init__(self, model, eps=0.03, a=0.001, u=1.0, random_star=True, distance=False, clip_min=-1.0, clip_max=1.0, flag_target=True):
        """
        :param model:       The victim model(Only support nn.Model)
        :param eps:         The upper bound of the disturbance
        :param a:           step size of each iteration
        :param a:           Coefficient of gradient momentum
        :param random_star: Whether to add random noise at the beginning
        :param distance:    Distance metric norm
        :param clip_min:    Lower bound of clip function
        :param clip_max:    upper bound of clip function
        :param flag_target: Whether there is a targeted attack
        """
        super(FaceMIFGSM, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.eps = eps
        self.a = a
        self.u = u
        # The number of iterations is limited by the eps and step size
        self.interate = int(self.eps/self.a)
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
        # Initialization gradient
        grad_ = ToCpuNumpy(torch.zeros(x.shape).type_as(x))

        for i in tqdm(range(self.interate)):
            # x_adv = x_new + pertubation
            embedding_new = self.model(x_new)
            embedding_new = embedding_new.squeeze()

            if self.flag_target:

                loss = -self.loss_func(target_embedding, embedding_new)
            else:
                loss = self.loss_func(embedding_true, embedding_new)

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            # x_new = x_new.squeeze()

            
            """ Core formula:
            g_t+1 = g_t * u + grad_t+1
            """
            grad_ = grad_ * self.u + ToCpuNumpy(x_new.grad)
            grad = np.sign(grad_)
            pertubation = grad * self.a

            x_new = x_new.cpu().detach().numpy() + pertubation
            x = x.cpu().detach().numpy()

            # clip
            x_new = np.clip(x_new, self.clip_min, self.clip_max)

            # pertubation after clip
            pertubation = x_new - x

            x = torch.Tensor(x)
            x_new = torch.Tensor(x_new).type_as(x).cuda()
            x_new.requires_grad = True

        x_adv = x_new.cpu().detach().numpy()
        pertubation = pertubation
        return x_adv, pertubation