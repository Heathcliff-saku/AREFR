import dlib
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


def face_alignment(faces,show=False):
    '''
    faces: num * width * height * channels ,value = 0~255, dtype = np.uint8,
    note: width must equal to height
    '''
    print(faces.shape)
    if len(faces.shape)==4 and faces.shape[3]==1:
        faces = faces.reshape(faces.shape[:-1]) # if gray, turns to num * width * height, no channel axis 如果是灰度图，去掉最后一维，否则predictor会报错
    num = faces.shape[0]

    faces_aligned = np.zeros(shape=faces.shape,dtype=np.uint8)

    predictor_path = "./shape_predictor_68_face_landmarks.dat" # dlib提供的训练好的68个人脸关键点的模型，网上可以下
    predictor = dlib.shape_predictor(predictor_path) # 用来预测关键点
    for i in range(num):
        img = faces[i]
        rec = dlib.rectangle(0,0,img.shape[0],img.shape[1])
        shape = predictor(np.uint8(img),rec) # 注意输入的必须是uint8类型
        order=[36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        if show:
            plt.figure()
            plt.imshow(img,cmap='gray')
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
                plt.scatter(x,y) # 可以plot出来看看效果，这里我只plot5个点
        eye_center =( (shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
#        print angle

        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0], img.shape[1])) # 进行放射变换，即旋转
        faces_aligned[i] = RotImg
    return faces_aligned # uint8


im_raw1 = cv2.imread('./1.png')
plt.figure() # plt是import matplotlib as plt，这里的输入最好是unint8
plt.imshow(im_raw1,cmap='gray')
im_raw1 = cv2.resize(im_raw1,(181,181))
#plt.figure()
imgs = np.zeros([2,181,181,3],dtype=np.uint8)
imgs[0] = im_raw1
print('size:',imgs.size)
faces_aligned = face_alignment(imgs,show=True)
#plt.figure()
plt.imshow(faces_aligned[0],cmap='gray')
plt.show()
