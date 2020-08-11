"""2.朴素贝叶斯分类识别

将12*33=396张图像按照训练集为70%，测试集为30%的比例随机划分，
再获取每张图像的骨骼和手部特征点的距离和角度
根据像素的特征分布情况进行图像分类分析。
"""

# -*- coding: utf-8 -*-
import os
from cv2 import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals import joblib

from pose_hand import getImgInfo
from yolo import YOLO
from pose.coco import general_coco_model

# ----------------------------------------------------------------------------------
# 第一步 切分训练集和测试集
# ----------------------------------------------------------------------------------

X = []  # 定义图像名称
Y = []  # 定义图像分类类标
# Z = [] #定义图像像素

path = 'D:/myworkspace/dataset/My_test/bagofwords'

for idx, labelname in enumerate(os.listdir(path)):
    if ".txt" not in labelname:
        f = os.path.join(path, labelname)
        for i, imgname in enumerate(os.listdir(f)):
            imgpath = os.path.join(f, imgname)
            X.append(imgpath)
            Y.append(labelname)

X = np.array(X)
Y = np.array(Y)

#随机率为100% 选取其中的30%作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# ----------------------------------------------------------------------------------
# 第二步 图像读取及转换为像素直方图
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


# # 训练集
# XX_train = []
# for i in X_train:
#     image = cv2.imread(i) 
#     hist,_= getImgInfo(image, pose_model, _yolo)

#     XX_train.append(hist)


# 测试集
XX_test = []
for i in X_test:
    image = cv2.imread(i)  
    hist,_ = getImgInfo(image, pose_model, _yolo)

    XX_test.append(hist)


# orb = open('D:/myworkspace/dataset/My_test/bagofwords/y_train.txt', 'w')
# for i, info in enumerate(y_train):
#     orb.write('\n'+str(info)+str(XX_train[i]))
# orb.close()
# orb = open('D:/myworkspace/dataset/My_test/bagofwords/y_test.txt', 'w')
# for i, info in enumerate(y_test):
#     orb.write('\n'+str(info)+str(XX_test[i]))
# orb.close()

# ----------------------------------------------------------------------------------
# 第三步 基于朴素贝叶斯的图像分类处理
# ----------------------------------------------------------------------------------
# 使用训练集训练模型
# clf = BernoulliNB().fit(XX_train, y_train)  # 伯努利贝叶斯分类器

os.chdir("model/")
# ticks = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) # 时间戳
# #joblib.dump(clf, ticks+"train_model.m") # 保存
# joblib.dump(clf, ticks+"train_model.pkl") # 保存

clf = joblib.load("train_model.pkl")
predictions_labels = clf.predict(XX_test)

#查看模型参数
print(clf.get_params())   # 参看模型的所有参数

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(clf, XX_train, y_train, cv=5,n_jobs=1)  # 采用5折交叉验证
# print(scores)
# # 平均得分和95%的置信区间
# print("Accuracy: %0.2f(+/-%0.3f)"%(scores.mean(), scores.std()*2))
# # 95%的置信区间在平均值两倍标准差之内

# 使用测试集预测结果
print(u'预测结果:')
print(predictions_labels)

# 生成文本型分类报告
print(u'算法评价:')  # 算法评价准确率（Precision）、召回率（Recall）和F值（F1-score）
print((classification_report(y_test, predictions_labels)))
# 生成字典型分类报告
report = classification_report(y_test, predictions_labels, output_dict=True)
# print(u'sun的字典型分类报告:')
# for key, value in report["sun"].items():
#     print(f"{key:10s}:{value:10.2f}")

# 输出前10张图片及预测结果
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
k = 0
while k < 10:
    # 读取图像
    print(X_test[k])
    image = cv2.imread(X_test[k])
    print(predictions_labels[k])
    # 显示图像
#    cv2.imshow("img", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     plt.figure(figsize=[5, 5])
#     plt.subplot(1, 1, 1)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.ylabel(u'%s'%str(predictions_labels[k]), fontsize=15)
#     plt.xlabel(u'%s'%str(X_test[k]))
#     plt.axis("off")
#     plt.show()
    k = k + 1