import os
import os.path as osp
from imutils import paths
import numpy as np
import dlib
import matplotlib.pyplot as plt

import cv2 as cv
from PIL import Image
import pickle
from config import Config as conf


def transform_clean_list(webface_directory, clean_list_path):
    """ 将txt转换为干净数据列表
    input：
         webface_directory：webface图像数据集地址
         clean_list_path：.txt文件地址
    outputs：
         cleaned_list:转换后的干净数据列表
    """
    with open(clean_list_path, encoding='utf-8') as f:
        cleaned_list = f.readlines()
    #cleaned_list = [p.replace('\\', '/') for p in cleaned_list]
    cleaned_list = [osp.join(webface_directory, p) for p in cleaned_list]
    return cleaned_list


def clean_data(webface_directory, cleaned_list):
    """ 删除不在干净列表的图像数据
    input：
         webface_directory：webface图像数据集地址
         cleaned_list：干净数据列表
    """
    cleaned_list = set([c.split()[0] for c in cleaned_list])
    for p in paths.list_images(webface_directory):
        if p not in cleaned_list:
            print(f"remove{p}")
            os.remove(p)


def PILImage2CV(PIL_img):
    # PIL Image转换成OpenCV格式
    img = cv.cvtColor(np.asarray(PIL_img), cv.COLOR_RGB2BGR)
    return img


def CVImage2PIL(cv_img):
    # OpenCV图片转换为PIL image
    img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2RGB))
    return img

def save_dict(dict):
    # 保存字典为pkl文件
    with open('feature_dict.pkl', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def load_dict():
    # 读取字典
    with open('feature_dict.pkl', 'rb') as f:
        return pickle.load(f)

def FaceDetector():
    if conf.face_detector == 'dlib_68':
        detector = dlib.get_frontal_face_detector()

    if conf.face_detector == 'opencv_haar':
        detector = cv.CascadeClassifier("./face_detect_feature/haarcascades/haarcascade_frontalface_default.xml")

def PlotSigleResult(adv_x, x, target):
   """
   绘制攻击结果的四宫格图，输入均为PIL格式的二维数组
   """
   plt.figure()
   plt.subplot(2, 2, 1)
   plt.imshow(adv_x, cmap='gray')
   plt.title("adv_img")

   plt.subplot(2, 2, 2)
   plt.imshow(x.cpu().detach().numpy(), cmap='gray')
   plt.title("origin_img")

   plt.subplot(2, 2, 4)
   pertubation = adv_x - x.cpu().detach().numpy()
   plt.title("pertubation")
   plt.imshow(pertubation, cmap='gray')

   plt.subplot(2, 2, 3)
   plt.title("Attack_target")
   plt.imshow(target.cpu().detach().numpy(), cmap='gray')
   plt.show()

def PlotSigleResult_phy(adv_x, x, target, patch):
   """
   绘制攻击结果的四宫格图，输入均为PIL格式的二维数组
   """
   plt.figure()
   plt.subplot(2, 2, 1)
   plt.imshow(adv_x, cmap='gray')
   plt.title("adv_img")

   plt.subplot(2, 2, 2)
   plt.imshow(x.cpu().detach().numpy(), cmap='gray')
   plt.title("origin_img")

   plt.subplot(2, 2, 4)
   pertubation = patch
   plt.title("pertubation")
   plt.imshow(pertubation, cmap='gray')

   plt.subplot(2, 2, 3)
   plt.title("Attack_target")
   plt.imshow(target.cpu().detach().numpy(), cmap='gray')
   plt.show()

if __name__ == '__main__':
    data = 'D:/CASIA-WebFace/'
    lst = './data/cleaned list.txt'
    cleaned_list = transform_clean_list(data, lst)
    #print(cleaned_list)
    clean_data(data, cleaned_list)