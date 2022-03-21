from sys import flags
from numpy.lib.function_base import average
from numpy.lib.polynomial import roots
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm


from Attack import FacePGD, FaceCW
from Attack.FGSM import FaceFGSM, FaceBIM, FaceMIFGSM
from model import FaceMobileNet
from config import Config as conf
from datetime import datetime
from utils import load_dict, PlotSigleResult


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def target_attack_get_root():
    """ 
    Split the input and target to get their root
    """
    pair_list = './data/lfw_test_pair_target_attack'
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    test_input_root = []
    test_target_root = []
    for pair in pairs:
        id1, id2, _ = pair.split()
        test_input_root.append('./data/lfw-align-128/'+id1)
        test_target_root.append('./data/lfw-align-128/'+id2)
    return test_input_root, test_target_root


def comput_distance(x1, x2, model):
    x1_embeding = model(x1).cpu().detach().numpy()
    x2_embeding = model(x2).cpu().detach().numpy().reshape(-1)
    distance = cosin_metric(x1_embeding, x2_embeding)
    return distance


def single_test(input_x_root, target_x_root, attack_method, model):
    input_x = Image.open(input_x_root)
    input_x = conf.test_transform(input_x)
    input_x = input_x[:, None, :, :]

    target_x = Image.open(target_x_root)
    target_x = conf.test_transform(target_x)
    target_x = target_x[:, None, :, :]

    x_adv, pertubation = attack_method.generate(input_x, target_x)
    x_adv_tensor = torch.Tensor(x_adv).to(torch.device("cuda"))
    input_x_tensor = input_x.to(torch.device("cuda"))
    target_x_tensor = target_x.to(torch.device("cuda"))

    xadv_x_distance = comput_distance(x_adv_tensor, input_x_tensor, model)
    xadv_target_distance = comput_distance(x_adv_tensor, target_x_tensor, model)

    if xadv_target_distance > xadv_x_distance:
        flag = True
    else:
        flag = False
    return flag, xadv_x_distance, xadv_target_distance

    
if __name__ == '__main__':
    # Loading victim model...
    victim_model = FaceMobileNet(conf.embedding_size).cuda()
    victim_model = nn.DataParallel(victim_model)
    state_dict = torch.load(conf.test_model, map_location='cuda')
    victim_model.load_state_dict(state_dict)
    victim_model.eval()

    # target-attack flow （for LFW evaluate）
    # first need Specify attack method

    # attack_method = FacePGD(victim_model, eps=0.03, a=0.001, interate=40, random_star=True, flag_target=True)
    # attack_method = FaceCW(victim_model)
    # attack_method = FaceFGSM(victim_model)
    # attack_method = FaceBIM(victim_model)
    attack_method = FaceMIFGSM(victim_model)

    # then get the data root
    test_input_roots, test_target_roots = target_attack_get_root()

    success = 0
    dis_target = []
    dis_input = []
    # for i in tqdm(range(len(test_target_roots))):
    for i in tqdm(range(1000)):
        input_root = test_input_roots[i]
        target_root = test_target_roots[i]
        flag, xadv_x_distance, xadv_target_distance = single_test(input_root, target_root, attack_method, victim_model)
        if flag:
            success += 1
        dis_target.append(xadv_target_distance)
        dis_input.append(xadv_x_distance)
    
    acc = success/1000
    average_dis_target = np.mean(np.array(dis_target))
    average_dis_input =  np.mean(np.array(dis_input))

    print('Attack success rate:\n', acc)  
    print('The average distance between adv_x and x:\n', average_dis_input)
    print('The average distance between adv_x and target:\n', average_dis_target)



