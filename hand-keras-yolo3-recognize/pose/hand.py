#!/usr/bin/python3
#!--*-- coding: utf-8 --*--
from cv2 import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from pose.data_process import distance,myAngle

class general_hand_model(object):
    def __init__(self, modelpath):
        # 指定采用的模型
        #   hand: 22 points（21个手势关键点，第22个点表示背景）
        #   COCO:   18 points（）
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        self.hand_num_points = 22
        self.hand_point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                                 [0, 5], [5, 6], [6, 7], [7, 8],
                                 [0, 9], [9, 10], [10, 11], [11, 12],
                                 [0, 13], [13, 14], [14, 15], [15, 16],
                                 [0, 17], [17, 18], [18, 19], [19, 20]]
        self.hand_net = self.get_hand_model(modelpath)

    """提取手势图像（在骨骼基础上定位左右手图片），handpose特征点，并可视化显示"""

    def get_hand_model(self, modelpath):

        prototxt = os.path.join(modelpath, "hand/pose_deploy.prototxt")
        caffemodel = os.path.join(
            modelpath, "hand/pose_iter_102000.caffemodel")
        hand_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return hand_model

    def getOneHandKeypoints(self, handimg):
        """hand手部关键点检测（单手）

        :param 手部图像路径，手部关键点
        :return points单手关键点坐标集合
        """
        img_height, img_width, _ = handimg.shape
        aspect_ratio = img_width / img_height

        inWidth = int(((aspect_ratio * self.inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(
            handimg, 1.0 / 255, (inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)

        self.hand_net.setInput(inpBlob)

        output = self.hand_net.forward()

        # vis heatmaps
        #self.vis_hand_heatmaps(handimg, output)

        #
        points = []
        for idx in range(self.hand_num_points):
            probMap = output[0, idx, :, :]  # confidence map.
            probMap = cv2.resize(probMap, (img_width, img_height))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        return points

    def getHandROI(self, imgfile, bonepoints):
        """hand手部感兴趣的区域寻找到双手图像

        :param 图像路径，骨骼关键点
        :return 左手关键点，右手关键点坐标集合,原始图片位置参数
        """
        img_cv2 = cv2.imread(imgfile)  # 原图像
        img_height, img_width, _ = img_cv2.shape
        rimg = img_cv2.copy()  # 图像备份
        limg = img_cv2.copy()
        # 以右手首为中心，裁剪长度为小臂长的图片
        if bonepoints[4] and bonepoints[3]:  # 右手
            h = int(distance(bonepoints[4], bonepoints[3]))  # 小臂长
            x_center = bonepoints[4][0]
            y_center = bonepoints[4][1]
            x1 = x_center-h
            y1 = y_center-h
            x2 = x_center+h
            y2 = y_center+h
            # print(x1,x2,x_center,y_center,y1,y2)
            if x1 < 0:
                x1 = 0
            if x2 > img_width:
                x2 = img_width
            if y1 < 0:
                y1 = 0
            if y2 > img_height:
                y2 = img_height
            rimg = img_cv2[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
        if bonepoints[7] and bonepoints[6]:  # 左手
            h = int(distance(bonepoints[7], bonepoints[6]))  # 小臂长
            x_center = bonepoints[7][0]
            y_center = bonepoints[7][1]
            x1 = x_center-h
            y1 = y_center-h
            x2 = x_center+h
            y2 = y_center+h
            #print(x1, x2, x_center, y_center, y1, y2)
            if x1 < 0:
                x1 = 0
            if x2 > img_width:
                x2 = img_width
            if y1 < 0:
                y1 = 0
            if y2 > img_height:
                y2 = img_height
            limg = img_cv2[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]

        return rimg, limg

    def getHandsKeypoints(self, rimg, limg):
        """双手图像分别获取特征点

        :param 图像路径，骨骼关键点
        :return 左手关键点，右手关键点坐标集合
        """
        # 分别获取手部特征点
        rhandpoints = self.getOneHandKeypoints(rimg)
        lhandpoints = self.getOneHandKeypoints(limg)
        # 显示
        self.vis_hand_pose(rimg, rhandpoints)
        self.vis_hand_pose(limg, lhandpoints)

        return rhandpoints, lhandpoints

    def vis_hand_heatmaps(self, handimg, net_outputs):
        """显示手势关键点热力图（单手）

        :param 图像路径，神经网络
        """
        plt.figure(figsize=[10, 10])

        for pdx in range(self.hand_num_points):
            probMap = net_outputs[0, pdx, :, :]
            probMap = cv2.resize(probMap, (handimg.shape[1], handimg.shape[0]))
            plt.subplot(5, 5, pdx+1)
            plt.imshow(cv2.cvtColor(handimg, cv2.COLOR_BGR2RGB))
            plt.imshow(probMap, alpha=0.6)
            plt.colorbar()
            plt.axis("off")
        plt.show()

    def vis_hand_pose(self, handimg, points):
        """显示标注手势关键点后的图像（单手）

        :param 图像路径，每只手检测关键点坐标
        :return 关键点连线图，关键点图
        """
        img_cv2_copy = np.copy(handimg)
        for idx in range(len(points)):
            if points[idx]:
                cv2.circle(
                    img_cv2_copy, points[idx], 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(img_cv2_copy, "{}".format(idx), points[idx], cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255), 1, lineType=cv2.LINE_AA)

        # Draw Skeleton
        for pair in self.hand_point_pairs:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(handimg, points[partA],
                         points[partB], (0, 255, 255), 2)
                cv2.circle(
                    handimg, points[partA], 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        plt.figure(figsize=[10, 10])
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(handimg, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
