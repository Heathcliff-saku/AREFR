"""
AREFR 物理对抗
使用patch攻击实现, 首先计算出一个mask M, 贴纸区域为1 非贴纸区域为0
设 干净样本为 x  攻击目标为x_t特征为y_t 对抗样本为x_adv

基本迭代形式为：

模仿攻击: x_adv (t+1) = (x_adv(t) - alpha*gradient(x_adv(t), y_t))  * M + x * (1-M)     x_adv(0) = x_target + 随机扰动
躲避攻击: x_adv (t+1) = (x_adv(t) - alpha*gradient(x_adv(t), y_t))  * M + x * (1-M)     x_adv(0) = x + 随机扰动

"""
import dlib
import cv2 as cv
from sympy import im
import torch 
import numpy as np
from imutils import face_utils
import torch.nn as nn
from tqdm import tqdm

from Attack.Loss import CosLoss

'加载人脸关键点检测器'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/Silvester/PycharmProjects/AREFR/shape_predictor_68_face_landmarks.dat')

'灰度图'
def Tensor2Numpy(x_tensor):
    return x_tensor.cpu().detach().numpy()

def Numpy2Tensor(x_numpy):
    return torch.from_numpy(x_numpy)

def BuiltMask(x, region):
    '''
    input:
        x:      干净样本 tensor
        region: 贴纸区域
    output:
        M: Mask
    '''
    print(x.shape)
    # x: ndarray [128, 128]
    boundary = detector(x, 1)

    for (index, rectangle) in enumerate(boundary):
        shape = predictor(x, rectangle)
        shape = face_utils.shape_to_np(shape)

        if region == 'glass':
            x1 = shape[17][0]
            x2 = shape[26][0]
            y1 = shape[19][1]
            y2 = shape[29][1]
        '''
        x1,y1                
         |----------------|
         |     eyes       |
         |----------------| x2,y2
        '''
        if region == 'mask':
            x1 = shape[5][0]
            x2 = shape[11][0]
            y1 = shape[30][1]
            y2 = shape[8][1]

    [H, W] = np.shape(x)

    '初始化掩膜Mask为0'
    Mask = np.zeros([H, W])
    for i in range(W):
        for j in range(H):
            if i>x1 and i<x2:
                if j>y1 and j<y2:
                    Mask[j, i] = 1
    return Mask

def PutPatch(x, x_, mask):
    '''
    input:
        x:      干净样本 tensor
        x_:     当前的加上对抗扰动后的对抗样本 x_adv(t) - alpha*gradient(x_adv(t), y_t) tensor
    output:
        x_adv_next: 下一轮的对抗样本 tensor
    '''

    x = Tensor2Numpy(x)
    x_adv_next = x_*mask + x*(1-mask)
    x_adv_next = Numpy2Tensor(x_adv_next)

    return x_adv_next


class PyhAttack(nn.Module):
    """ Adversarial physical Attack for AREFR: -- v 1.0
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
    def __init__(self, model, eps=100000, a=0.1, interate=500, random_star=True, distance=False, clip_min=-1.0, clip_max=1.0, flag_target=True):
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
        super(PyhAttack, self).__init__()
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

    def generate(self, x, target, x_for_mask):
        """
        :param x:       Clean sample's image of one person  
        :param target:  The img of the attack target people (Same as above)
        :return:        adv_x
        """

        if self.flag_target:
            if self.random_star:
                x_new = target + torch.Tensor(np.random.uniform(-0.03, 0.03, target.shape)).type_as(target)
                x_new.requires_grad = True
            else:
                x_new = target
        else:
            if self.random_star:
                x_new = x + torch.Tensor(np.random.uniform(-0.03, 0.03, x.shape)).type_as(x)
            else:
                x_new = x
        

        embedding_true = self.model(x).squeeze()
        embedding_target = self.model(target).squeeze()

        # x : tensor [1, 1, 128, 128]
        Mask_x = BuiltMask(x_for_mask, 'mask')
        # mask : tensor [1, 1, 128, 128]

        for i in range(self.interate):
            # x_adv = x_new + pertubation
            embedding_new = self.model(x_new)
            embedding_new = embedding_new.squeeze()

            if self.flag_target:
                loss = -self.loss_func(embedding_target, embedding_new)
            else:
                loss = self.loss_func(embedding_true, embedding_new)
            
            print('loss: ', loss)

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            # x_new = x_new.squeeze()

            grad = x_new.grad.cpu().detach().numpy()
            grad = np.sign(grad)

            if i % 100 == 0:
                self.a = self.a * 0.9
            pertubation = grad * self.a

            x_new = x_new.cpu().detach().numpy() + pertubation
            x = x.cpu().detach().numpy()

            # clip
            x_new = np.clip(x_new, x - self.eps, x + self.eps)
            x_new = np.clip(x_new, self.clip_min, self.clip_max)

            x = torch.Tensor(x)
            x_new = PutPatch(x, x_new, Mask_x).type_as(x).cuda()
            x_new.requires_grad = True

        x_adv = x_new.cpu().detach().numpy()
        patch = x_adv * Mask_x
        pertubation = x_adv * Mask_x
        return x_adv, pertubation, patch


def test():
    x = cv.imread("C:/Users/Silvester/PycharmProjects/AREFR/data/facebank/Zhang_Qiyang/1.jpg")
    x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    x_ = cv.imread("C:/Users/Silvester/PycharmProjects/AREFR/data/facebank/rsw/1.jpg")
    x_ = cv.cvtColor(x_, cv.COLOR_BGR2GRAY)

    x = torch.from_numpy(x)
    x_ = torch.from_numpy(x_)

    Mask = BuiltMask(x, region='mask')
    x_adv_next = PutPatch(x, x_, mask=Mask)

    cv.imshow('x_adv_next', x_adv_next/255)
    cv.waitKey(0)