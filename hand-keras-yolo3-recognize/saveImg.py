import os
import time
import numpy as np
from pose.coco import general_coco_model
from pose.hand import general_hand_model
from pose.data_process import getBoneInformation, getHandsInformation
from pose.hand_fD import hand_fourierDesciptor
from yolo import YOLO
from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import re


def bgImg_save(img_path,save_path):
    # 读取图像
    print("[INFO]",img_path)
    image = cv2.imread(img_path)

    # 图像像素大小一致
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    # pose骨骼
    start = time.time()
    bone_points = pose_model.getBoneKeypoints(img)  # 2.骨骼关键点
    lineimage,dotimage,black_np = pose_model.vis_bone_pose(img, bone_points)  # 骨骼连线图、标记图显示cv2格式
    list1 = getBoneInformation(bone_points)  # 3.骨骼特征

    # yolo手
    image = Image.open(img_path)
    lineimage = Image.fromarray(cv2.cvtColor(lineimage,cv2.COLOR_BGR2RGB))# cv2图片转PIL
    black_np = Image.fromarray(cv2.cvtColor(black_np,cv2.COLOR_BGR2RGB))
    line_image,labelinfo,hand_ROI_PIL = _yolo.detect_image(image,black_np) # (原图，lineimage线图,黑幕图)
    print("[INFO]Model predicts time: ", time.time() - start)


    # info = []
    # for i in range(len(list1)):
    #     info.append(list1[i])
    # for j in range(len(labelinfo)):
    #     info.append(labelinfo[j])

    # print(labelinfo)
    line_image.save(save_path)


#---------------------------------
#  1.加载模型
#---------------------------------

# coco
modelpath = "model/"
start = time.time()
pose_model = general_coco_model(modelpath)  # 1.加载模型

print("[INFO]Pose Model loads time: ", time.time() - start)

# yolo
start = time.time()
_yolo = YOLO() # 1.加载模型

print("[INFO]yolo Model loads time: ", time.time() - start)

imgpath='D:/myworkspace/dataset/My_test/bagofwords/you/you_1_32.png'

bgImg_save(imgpath,'')

'''
X = []  # 定义图像名称
Y = []  # 定义图像分类类标
# Z = [] #定义图像像素

path = 'D:/myworkspace/dataset/My_test/dataset/hand_classification/'
savepath = 'D:/myworkspace/dataset/My_test/dataset/hand_background_classification/'

for idx, labelname in enumerate(os.listdir(path)):
    if ".txt" not in labelname:
        f = os.path.join(path,labelname)
        s = os.path.join(savepath,labelname)
        if not os.path.exists(s):
            os.makedirs(s)
        for i, imgname in enumerate(os.listdir(f)):
            imgpath = os.path.join(f,imgname)
            #X.append(imgpath)
            #Y.append(labelname)
            save_path = os.path.join(s,imgname)
            bgImg_save(imgpath,save_path)

            #num = re.findall(".*_(.*)_.*", imgname)
            # if num[0]=="7":
            #     bgImg_save(imgpath,save_path)
'''