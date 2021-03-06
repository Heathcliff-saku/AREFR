import cv2 as cv
import os
import os.path as osp
from pathlib import Path
from datetime import datetime

from sympy import im
from config import Config as conf
import sys
import dlib
from imutils import face_utils

from face_detect import FaceDetect
"""
实现从摄像头获取面部图像 实现自动裁剪、自动保存的功能
1、用于制作自己的数据集
2、用于后期实时检测

已实现功能：
从摄像头获取面部图像，并自动裁剪保存
从图片路径获取面部图像，并自动保存
"""

def SavePicture(img):
    """ 保存图像功能 """
    name = conf.name
    data_path = Path('data')
    save_path = data_path/'facebank'/name
    if not osp.exists(save_path):
        os.makedirs(save_path)
    idx = 0
    for root, dirs, filenames in os.walk(save_path):
        for file in filenames:
            idx += 1
    cv.imwrite(str(save_path / '{}.jpg'.format(str(idx+1))), img)
    print('one picture has saved')



def OpenCam(window_name, viedo_id):
    # face_detector = cv.CascadeClassifier("./face_detect_feature/lbpcascades/lbpcascade_frontalcatface.xml")
    # face_detector = cv.CascadeClassifier("./face_detect_feature/haarcascades/haarcascade_frontalface_default.xml")

    cv.namedWindow(window_name)
    cap = cv.VideoCapture(viedo_id)
    cap.set(3, 1920)  # cap width size
    cap.set(4, 1080)  # cap length size
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        """ put text """
        cv.putText(frame, 'put q to exit, t to save picture', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
        cv.imshow(window_name, frame)
        c = cv.waitKey(1)  # 0.01ms一帧，返回键盘的按键
        if c & 0xFF == ord('q'):  # 按下q退出
            break
        if c & 0xFF == ord('t'): # 按下t拍照保存
            try:
                face_img = FaceDetect((128, 128), frame)
                SavePicture(face_img)
            except:
                print('no face detected, please try again')

    cap.release()
    cv.destroyWindow(window_name)
    print('cam has closed')

if __name__ == '__main__':
    if conf.use_cam:
        print('cam opening ...')
        window_name = 'take your face picture'
        OpenCam(window_name, 0)
    else:
        face_img = cv.imread(conf.face_img)
        face_img = FaceDetect((128, 128), face_img)
        SavePicture(face_img)


