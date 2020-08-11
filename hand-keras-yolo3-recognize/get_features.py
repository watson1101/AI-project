# 从人脸图像文件中提取人脸特征存入 CSV
# Features extraction from images and save into features_all.csv

# return_128d_features()          获取某张图像的128D特征
# compute_the_mean()              计算128D特征均值

from cv2 import cv2 as cv2
import os
import csv
import numpy as np
import time

from pose_hand import getImgInfo
from yolo import YOLO
from pose.coco import general_coco_model

# ----------------------------------------------------------------------------------
# 第一步 读取标签Y和图片路径X
# ----------------------------------------------------------------------------------

X = []  # 定义图像名称
Y = []  # 定义图像分类类标
# Z = [] #定义图像像素

path = 'D:/myworkspace/dataset/My_test/dataset/test'

for idx, labelname in enumerate(os.listdir(path)):
    if ".txt" not in labelname:
        f = os.path.join(path, labelname)
        for i, imgname in enumerate(os.listdir(f)):
            imgpath = os.path.join(f, imgname)
            X.append(imgpath)
            Y.append(labelname)

X = np.array(X)
Y = np.array(Y)


# ----------------------------------------------------------------------------------
# 第二步 识别infolist
# ----------------------------------------------------------------------------------
# coco
modelpath = "model/"
start = time.time()
pose_model = general_coco_model(modelpath)  # 1.加载模型
print("[INFO]Pose Model loads time: ", time.time() - start)
# yolo
start = time.time()
_yolo = YOLO()  # 1.加载模型
print("[INFO]yolo Model loads time: ", time.time() - start)


infolist = []
for i in X:
    hist = getImgInfo(i, pose_model, _yolo) # 识别
    infolist.append(hist)


# ----------------------------------------------------------------------------------
# 第三步 存储信息docs/feature/features_all.csv
# ----------------------------------------------------------------------------------
# 路径存储到txt
orb = open('D:/myworkspace/dataset/My_test/bagofwords/y_train.txt', 'w')
for i, img_path in enumerate(X):
    orb.write(img_path)
    #orb.write('\n'+str(info)+str(infolist[i]))
orb.close()

# 特征存储到 csv
with open("docs/feature/features_all.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for i in infolist:    
        writer.writerow(i)
    print("所有录入手语特征数据存入 / Save all the features of sign registered into: docs/feature/features_all.csv")
