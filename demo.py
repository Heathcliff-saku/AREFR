import imp
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datetime import datetime
import cv2 as cv
from tqdm import tqdm
import dlib
from imutils import face_utils


from config import Config as conf
from model import FaceMobileNet
from utils import PILImage2CV, CVImage2PIL  # 图像格式转换
from utils import save_dict, load_dict

from torchvision.datasets import ImageFolder
from face_detect import GetFaceCoord, FaceDetect

""" 实现人脸识别功能
1、计算facebank(用于搜索比对的标准人脸)中的脸部特征，并保存为特征库
2、将待检测的人脸传入模型，得到特征
3、对每一个特征库中的图像，利用余弦距离度量向量距离，取距离最贴近的特征库，返回标签和面部图像 

已实现功能
√ 传入图片进行比对识别
√ 传入图片无需指定大小，自动裁剪
√ 提前保存特征向量，只需调用即可
√ 使用摄像头进行识别，将结果实时 展现 在屏幕

更新
2022.2.16 实现视频实时计预测（对cpu能力要求较高）

 """


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def make_dict(facebank_root, model, transform, device):
    """ 遍历facebank 制作特征字典 {face_feature: -- label:} """

    face_root = []  # 存放图像路径的list
    labels = []  # 存放对应标签的list
    imgs = []   # 存放图像矩阵的list

    """ 生成路径和标签的list """
    names = os.listdir(facebank_root)
    for name in names:
        faces = os.listdir(facebank_root + '/' + name)
        for face in faces:
            face_root.append(facebank_root + '/' + name + '/' + face)
            labels.append(name)

    """ 制作数据集送入网络 """
    for img in face_root:
        img = Image.open(img)
        img = transform(img)
        imgs.append(img)
    data = torch.cat(imgs, dim=0)
    data = data[:, None, :, :]
    data = data.to(device)
    model = model.to(device)
    with torch.no_grad():
        features = model(data)  # 存放对应特征的list
    feature_dict = {feature: label for (feature, label) in zip(features, labels)}
    save_dict(feature_dict)
    return feature_dict


def picture_recognition():
    begin_time = datetime.now()
    """ 加载模型 """
    model = FaceMobileNet(conf.embedding_size)
    model = nn.DataParallel(model)
    state_dict = torch.load(conf.test_model, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()

    # face_detector = cv.CascadeClassifier("./face_detect_feature/lbpcascades/lbpcascade_frontalcatface.xml")
    # face_detector = cv.CascadeClassifier("./face_detect_feature/haarcascades/haarcascade_frontalface_default.xml")
    # face_detector = dlib.get_frontal_face_detector()

    if conf.update_feature_dict:
        """ 加载特征字典 """
        feature_dict = make_dict(conf.facebank_root, model, conf.test_transform, device=conf.device)
    else:
        """ 直接读取特征字典 """
        feature_dict = load_dict()

    """ 计算测试图像的feature """
    input_img = Image.open(conf.input_root)

    input_img = PILImage2CV(input_img)  # 转化为cv格式的图像方便进行检测
    input_img = FaceDetect(size=(128, 128), img=input_img)
    input_img = CVImage2PIL(input_img)  # 转化回PIL格式的图像

    input_img = conf.test_transform(input_img)
    input_img = input_img[:, None, :, :]
    input_img = input_img.to(conf.device)
    model = model.to(conf.device)

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

    print('pred_name:', pred_name)
    print('similarity:', best_score)
    print('recognition use time:', over_time - begin_time)
    print("all score:", scores)


def video_recognition(window_name):
    """ 加载模型 """
    print('load model ... ')
    model = FaceMobileNet(conf.embedding_size)
    model = nn.DataParallel(model)
    state_dict = torch.load(conf.test_model, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()

    # face_detector = cv.CascadeClassifier("./face_detect_feature/lbpcascades/lbpcascade_frontalcatface.xml")
    # face_detector = cv.CascadeClassifier("./face_detect_feature/haarcascades/haarcascade_frontalface_default.xml")

    if conf.update_feature_dict:
        """ 加载特征字典 """
        feature_dict = make_dict(conf.facebank_root, model, conf.test_transform, device=conf.device)
    else:
        """ 直接读取特征字典 """
        feature_dict = load_dict()

    """ 打开摄像头 """
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(0)
    cap.set(3, 1920)  # cap width size
    cap.set(4, 1080)  # cap length size
    pred_name = 'initializing....'
    best_score = 'initializing....'
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        """ put text """
        cv.putText(frame, 'Face Recognition System for AREFR  author:rsw', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
        cv.putText(frame, 'Press t to Recognition, q to exit', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
        cv.putText(frame, 'Top-1 Result: ' + pred_name, (10, 500), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 55, 255), 2)
        cv.putText(frame, 'similarity: ' + str(best_score), (10, 550), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 55, 255), 2)
        try:
            x, y, w, h = GetFaceCoord(frame)
            cv.rectangle(frame, (x, y), (x+w, y+h), (55, 255, 155), 2)
            cv.putText(frame, pred_name, (x,y), (55, 255, 155), 2)
        except:
            print('no face detected, please try again')

        cv.imshow(window_name, frame)
        c = cv.waitKey(1)  # 0.01ms一帧，返回键盘的按键
        if conf.is_realtime:
            try:
                input_img = FaceDetect(size=(128, 128), img=frame)
                input_img = CVImage2PIL(input_img)  # 转化回PIL格式的图像
                input_img = conf.test_transform(input_img)
                input_img = input_img[:, None, :, :]
                input_img = input_img.to(conf.device)
                model = model.to(conf.device)

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
                #print('pred_name:', pred_name)
                #print('similarity:', best_score)
                #print("all score:", scores)

            except:
                print('no face detected, please try again')

        if c & 0xFF == ord('q'):  # 按下q退出
            break
        if c & 0xFF == ord('t'):  # 执行单次预测
            try:
                input_img = FaceDetect(size=(128, 128), img=frame)
                input_img = CVImage2PIL(input_img)  # 转化回PIL格式的图像
                input_img = conf.test_transform(input_img)
                input_img = input_img[:, None, :, :]
                input_img = input_img.to(conf.device)
                model = model.to(conf.device)

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
                print('pred_name:', pred_name)
                print('similarity:', best_score)
                print("all score:", scores)

            except:
                print('no face detected, please try again')

    cap.release()
    cv.destroyWindow(window_name)
    print('cam has closed')



if __name__ == '__main__':

    window_name = 'Face Recognition System'

    if conf.demo_use_cam:
        video_recognition(window_name)
    else:
        picture_recognition()


