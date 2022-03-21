import torch
import torchvision.transforms as T
import cv2 as cv
import dlib
from argparse import ArgumentParser

"""
配置相关路径 预处理超参数
"""

class Config:
    """ 数据集 """
    train_root = 'D:/CASIA-WebFace'
    test_root = './data/lfw-align-128'
    test_list = './data/lfw_test_pair.txt'
    # test_model = './checkpoints/mobile-arcface-epoch59.pth'
    test_model = './checkpoints/24.pth'

    """ 数据处理 """
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),  # 随机水平翻转
        T.Resize((144, 144)),   # 放缩
        T.RandomCrop(input_shape[1:]),  # 随机裁剪回input_size
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    """ 数据集加载 超参数 """
    parser = ArgumentParser(description='Process some integers.')
    
    train_batch_size = 8
    test_batch_size = 60
    pin_memory = False  # if memory is large, set it True for speed
    num_workers = 4

    """ 模型 超参数"""
    backbone = 'resnet'  # ['resnet', 'fmobile']
    metric = 'arcface'  # ['cosface', 'arcface']
    embedding_size = 512
    drop_ratio = 0.5

    """ 训练 """
    epoch = 60
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 0.1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss'  # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = 'checkpoints'  # the directory which put weight

    """ 人脸检测器 """
    face_detector = 'opencv_haar'  #[opencv_haar, dilb_68]


    """ 拍照程序制作人脸数据 """
    use_cam = False  # 采集方式 true为使用摄像头，False为传入照片
    face_img = './sample/zqy6.jpg'  # 采集的图像
    name = 'test'

    """ demo程序识别人脸 """
    demo_use_cam = True
    facebank_root = './data/facebank'
    update_feature_dict = True  # 是否更新特征字典（如果加入了新的人物）
    is_realtime = True  # 是否进行实时预测（若cpu性能较高）


    input_root = './sample/rsw_.png'  # 需要识别的人脸路径（demo用）




