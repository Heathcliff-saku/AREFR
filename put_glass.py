
"""
将眼镜渲染至面部图像, 返回带有眼镜的图像

2022.1.18: 更新— 使得配件可根据面部倾角进行适配性的旋转
           使用opencv仿射变换实现
           目前口罩部分 还存在一些问题 暂不支持适应性旋转 需要保持正面角度

2022.3.3: 更新 
          口罩部分 对面部首先采用仿射变换进行人脸矫正
"""

from pickletools import float8
from turtle import begin_poly
import matplotlib.pyplot as plt
import dlib
import imutils
import cv2 as cv
from imutils import face_utils
import math
import numpy as np


def overlay_image_alpha(img, img_overlay, begin_y, pos, alpha_mask):
    x, y = pos
    # Image ranges
    y1, y2 = max(0, y-int(begin_y)), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]  # alpha 为像素点的归一化系数 越黑部分越小
    alpha_inv = 1.0 - alpha
    for c in range(channels):
        for i in range(img_overlay.shape[0]):
            for j in range(img_overlay.shape[1]):
                if img_overlay[y1o+i, x1o+j, c] != 0:
                    # 若需要配件的浅色部分和皮肤进行叠加融合
                    # img[y1+i, x1+j, c] = img_overlay[y1o+i, x1o+j, c] * alpha_inv[y1o+i, x1o+j] + img[y1+i, x1+j, c] * alpha[y1o+i, x1o+j]
                    img[y1+i, x1+j, c] = img_overlay[y1o+i, x1o+j, c]

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    begin_y = w * sin
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv.warpAffine(image, M, (nW, nH),borderValue=(0, 0, 0)), begin_y
    # borderValue 缺省，默认是黑色（0, 0 , 0）



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.imread("./sample/rsw.jpg")
frame = cap.copy()
# loading the image and perform some operations on it

face_Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

boundary = detector(face_Gray, 1)
#     x_cor, y_cor = 0,0
#     x2, x1 = 0,0
#     y1,y2 = 0,0
#     slope = 0
#     z = 0
#     angle = 0
for (index, rectangle) in enumerate(boundary):
    shape = predictor(face_Gray, rectangle)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face_utils.rect_to_bb(rectangle)

    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.putText(frame, "Face {}".format(index + 1), (x - 10, y - 10), cv.FONT_ITALIC, 0.7, (0, 0, 255), 2)

    for (x, y) in shape:
        cv.circle(frame, (x, y), 2, (0, 0, 0), -1)

    x1 = shape[0][0]
    x2 = shape[16][0]
    y1 = shape[19][1]
    y2 = shape[28][1]

    slope = (shape[16][1]-shape[0][1])/(x2-x1)
    angle = math.degrees(math.atan(slope))

    s_img = cv.imread('./sample/glass1.png')
    #s_img = imutils.rotate(s_img, -angle)
    length = int(math.sqrt((x2-x1)**2 + (shape[16][1]-shape[0][1])**2))
    width = int(s_img.shape[0] * length / s_img.shape[1])
    s_img = cv.resize(s_img, (length, width)) 
    s_img, begin_y = rotate_bound_white_bg(s_img, angle)

    overlay_image_alpha(frame, s_img, begin_y, (shape[0][0], shape[17][1]), s_img[:, :, 2] / 255.0)

width = cap.shape[1]
height = cap.shape[0]
ratio = float(width) / float(height)
# print(int(500*ratio) )
new_height = 900
cap = cv.resize(cap, (int(new_height * ratio), new_height))
cv.imshow("original image", cap)

frame = cv.resize(frame, (int(new_height * ratio), new_height))
cv.imshow("detected face", frame)

k = cv.waitKey(0)
