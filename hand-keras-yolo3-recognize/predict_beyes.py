# -*- coding: utf-8 -*-
import os
from cv2 import cv2
import time
import numpy as np

from sklearn.externals import joblib

from pose_hand import getImgInfo
from yolo import YOLO
from pose.coco import general_coco_model


# coco
modelpath = "model/"
start = time.time()
pose_model = general_coco_model(modelpath)  # 1.加载模型
print("[INFO]Pose Model loads time: ", time.time() - start)
# yolo
start = time.time()
_yolo = YOLO()  # 1.加载模型
print("[INFO]yolo Model loads time: ", time.time() - start)

path = "D:/myworkspace/JupyterNotebook/hand-keras-yolo3-recognize/docs/wangyu_hand_img/"

X_test = [path+"movehouse_37.jpg",path+"movehouse_65.jpg"]

# 测试集
XX_test = []
for i in X_test:
    image = cv2.imread(i)  
    hist,_ = getImgInfo(image, pose_model, _yolo)

    XX_test.append(hist)

clf = joblib.load("model/train_model.pkl")
predictions_labels = clf.predict(XX_test)



# 使用测试集预测结果
print(u'预测结果:')
print(predictions_labels)

