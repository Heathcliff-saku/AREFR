from sympy import im
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os


from Attack import FacePGD
from Attack import FaceCW
from Attack.FGSM import FaceFGSM, FaceBIM, FaceMIFGSM
from Attack import Evolutionary
from Attack import AdvGlass

from model import FaceMobileNet
from config import Config as conf
from datetime import datetime
from utils import load_dict, PlotSigleResult

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def picture_recognition(model, input_img):
    begin_time = datetime.now()

    """ 直接读取特征字典 """
    feature_dict = load_dict()
    with torch.no_grad():
        input_feature = model(input_img)
    best_score = 0
    scores = []
    for feature in feature_dict.keys():
        input_feature = input_feature.cpu().reshape(-1)
        feature_ = feature.cpu().numpy()
        score = cosin_metric(feature_, input_feature)
        scores.append(score)
        if score > best_score:
            best_score = score
            pred_name = feature_dict.get(feature)
        else:
            continue
    over_time = datetime.now()
    use_time = over_time - begin_time

    return pred_name, best_score, scores


def digtal_single_test(input_x_root, target_x_root, attack_method, model):
    input_x = Image.open(input_x_root)
    input_x = conf.test_transform(input_x)
    input_x = input_x[:, None, :, :]

    target_x = Image.open(target_x_root)
    target_x = conf.test_transform(target_x)
    target_x = target_x[:, None, :, :]

    x_adv, pertubation = attack_method.generate(input_x, target_x)
    x_adv_tensor = torch.Tensor(x_adv).to(torch.device("cuda"))
    input_x_tensor = input_x.to(torch.device("cuda"))

    pred_name, best_score, scores = picture_recognition(model, input_x_tensor)
    pred_name_adv, best_score_adv, scores_adv = picture_recognition(model, x_adv_tensor)

    print('Before Attack:\n')
    print('pred_name:', pred_name)   
    print('similarity:', best_score)
    print("all score:", scores)

    print('Affter Attack:\n')
    print('pred_name:', pred_name_adv)
    print('similarity:', best_score_adv)
    print("all score:", scores_adv)

    if pred_name != pred_name_adv:
        print('Success Attack !')
    else:
        print('failed Attack')

    PlotSigleResult(x_adv[0, 0, :, :], input_x[0, 0, :, :], target_x[0, 0, :, :])

def physical_single_test(input_x_root, target_x_root, attack_method, model):
    x_batch = torch.zeros(size=(5, 1, 128, 128))
    n = 0
    for img in os.listdir(input_x_root):
        input_x = Image.open(input_x_root + img)
        input_x = conf.test_transform(input_x)
        # input_x = input_x[:, None, :, :]
        x_batch[n, :, :, :] = input_x
        n += 1

    target_x = Image.open(target_x_root)
    target_x = conf.test_transform(target_x)
    target_x = target_x[:, None, :, :]

    x_adv, pertubation = attack_method.generate(x_batch, target_x)
    # x_adv.shape = (5, 1, 128, 128)
    x_adv = x_adv[0, :, :, :]
    x_adv = x_adv[None, :, :, :]
    x_adv_tensor = torch.Tensor(x_adv).to(torch.device("cuda"))

    input_x = x_batch[0, :, :, :]
    input_x = input_x[None, :, :, :]
    input_x_tensor = input_x.to(torch.device("cuda"))

    pred_name, best_score, scores = picture_recognition(model, input_x_tensor)
    pred_name_adv, best_score_adv, scores_adv = picture_recognition(model, x_adv_tensor)

    print('Before Attack:\n')
    print('pred_name:', pred_name)   
    print('similarity:', best_score)
    print("all score:", scores)

    print('Affter Attack:\n')
    print('pred_name:', pred_name_adv)
    print('similarity:', best_score_adv)
    print("all score:", scores_adv)

    if pred_name != pred_name_adv:
        print('Success Attack !')
    else:
        print('failed Attack')

    PlotSigleResult(x_adv[0, 0, :, :], input_x[0, 0, :, :], target_x[0, 0, :, :])



if __name__ == '__main__':
    # Loading victim model...
    victim_model = FaceMobileNet(conf.embedding_size).cuda()
    victim_model = nn.DataParallel(victim_model)
    state_dict = torch.load(conf.test_model, map_location='cuda')
    victim_model.load_state_dict(state_dict)
    victim_model.eval()

    mode = 'digtal' # 'digtal' or 'Physical'

    if mode == 'digtal':
        """
        digtal attack flow
        you first need Specify attack method
        """
        attack_method = FacePGD(victim_model)
        # attack_method = FaceCW(victim_model)
        # attack_method = FaceFGSM(victim_model)
        # attack_method = FaceBIM(victim_model)
        # attack_method = FaceMIFGSM(victim_model)
        # attack_method = Evolutionary(victim_model)

        """ then give the root of input and attack target """
        input_x_root = './data/facebank/rsw/1.jpg'  # true: rsw
        target_x_root = './data/facebank/Yao_Ming/Yao_Ming_0001.jpg'  # target: zqy
        digtal_single_test(input_x_root, target_x_root, attack_method, victim_model)

    else:
        """
        Physical environment attack flow
        you first need Specify attack method
        """
        attack_method = AdvGlass(victim_model)

        """ 
        then give the root of input and attack target
        input: a batch of image [n, 1, 128, 128]
        target: one image of target [1, 128, 128]
        """
        input_x_root = './data/facebank/rsw/'
        target_x_root = './data/facebank/Yao_Ming/Yao_Ming_0001.jpg'

        # use single_test() to generate one attack result

        physical_single_test(input_x_root, target_x_root, attack_method, victim_model)
    
    