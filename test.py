import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import Config as conf
from model import FaceMobileNet
from collections import OrderedDict

""" test model by LFW dataset """

def unique_image(pair_list) -> set:
    """ 返回列表的不重复图片路径 """
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique

def group_image(images: set, batch) ->list:
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch+i, size)
        res.append(images[i:end])
    return res

def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)
    data = data[:, None, :, :]
    return data

def featurize(images: list, transform, net, device) -> dict:
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
        # print(features.size())
    res = {img: feature for (img, feature) in zip(images, features)}
    return res

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    print(y_score)
    print(y_true)
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th

def compute_accuracy(feature_dict, pair_list, test_root):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in tqdm(pairs, ascii=True, total=len(pairs)):
        img1, img2, label = pair.split()
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)

        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold

def testing():
    model = FaceMobileNet(conf.embedding_size)
    # 自己训练的权重要把下面这个加上
    model = nn.DataParallel(model)

    state_dict = torch.load(conf.test_model, map_location='cuda')
    
    new_state_dict = OrderedDict()
    for key, param in state_dict.items():
        name = key
        new_state_dict[name] = param

    model.load_state_dict(new_state_dict)
    model.eval()

    images = unique_image(conf.test_list)
    images = [osp.join(conf.test_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)
    feature_dict = dict()
    print('compute feature dict...')
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d)
        # print(len(feature_dict))
    accuracy, threshold = compute_accuracy(feature_dict, conf.test_list, conf.test_root)

    print(
        f"Test Model: {conf.test_model}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )


if __name__ == '__main__':
    testing()
