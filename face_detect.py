"""
人脸检测接口, 支持demo和take_picture的人脸检测步骤
提供 dlib_68、haar等检测模型
"""
import cv2 as cv
from config import Config as conf
import dlib
from imutils import face_utils


if conf.face_detector == "opencv_haar":
    detector = cv.CascadeClassifier("./face_detect_feature/haarcascades/haarcascade_frontalface_default.xml")
else:
    detector = dlib.get_frontal_face_detector()


def GetFaceCoord(img, face_detector=detector):
    """ 获取图像内人脸位置的坐标值 """
    if conf.face_detector == "opencv_haar":
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray_img, 1.2, 5)
        x = faces[0, :][0]
        y = faces[0, :][1]
        w = faces[0, :][2]
        h = faces[0, :][3]
        return x, y, w, h

    else:
        face_Gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        boundary = face_detector(face_Gray, 1)
        (x, y, w, h) = face_utils.rect_to_bb(boundary[0])
        return x, y, w, h


def FaceDetect(size, img):
    """ 实现人脸检测和裁剪 """
    if conf.face_detector == "opencv_haar":
        face_detector = face_detector = cv.CascadeClassifier("./face_detect_feature/haarcascades/haarcascade_frontalface_default.xml")
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        face = face_detector.detectMultiScale(gray_img, 1.2, 5)
        print(face[0, :])
        x = face[0, :][0]
        y = face[0, :][1]
        w = face[0, :][2]
        h = face[0, :][3]
        img = img[y:y+h, x:x+w]
        face_img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)
        return face_img
    else:
        face_detector = dlib.get_frontal_face_detector()
        face_Gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        boundary = face_detector(face_Gray, 1)
        (x, y, w, h) = face_utils.rect_to_bb(boundary[0])
        img = img[y:y+h, x:x+w]
        face_img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)
        return face_img
