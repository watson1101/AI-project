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

class general_coco_model(object):
    def __init__(self, modelpath):
        # 指定采用的模型
        #   hand: 22 points（21个手势关键点，第22个点表示背景）
        #   COCO:   18 points（）
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        self.pose_net = self.general_coco_model(modelpath)

    """提取骨骼特征点，并可视化显示"""

    def general_coco_model(self, modelpath):
        """COCO输出格式：
                鼻子-0, 脖子-1，右肩-2，右肘-3，右手腕-4，左肩-5，左肘-6，左手腕-7，右臀-8，右膝盖-9，右脚踝-10，
                左臀-11，左膝盖-12，左脚踝-13，右眼-14，左眼-15，有耳朵-16，左耳朵-17，背景-18.
        """
        self.points_name = {
            "Nose": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7,
            "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13,
            "REye": 14, "LEye": 15,
            "REar": 16, "LEar": 17,
            "Background": 18}
        self.bone_num_points = 18
        self.bone_point_pairs = [[1, 0], [1, 2], [1, 5],
                                 [2, 3], [3, 4], [5, 6],
                                 [6, 7], [1, 8], [8, 9],
                                 [9, 10], [1, 11], [11, 12],
                                 [12, 13], [0, 14], [0, 15],
                                 [14, 16], [15, 17]]
        prototxt = os.path.join(
            modelpath, "pose_coco/pose_deploy_linevec.prototxt")
        caffemodel = os.path.join(
            modelpath, "pose_coco/pose_iter_440000.caffemodel")
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model

    def getBoneKeypoints(self, img_cv2):
        """COCO身体关键点检测

        :param 图像路径
        :return 关键点坐标集合
        """
        #img_cv2 = cv2.imread(imgfile)
        
        img_height, img_width, _ = img_cv2.shape
        # 读取图像并生成输入blob
        inpBlob = cv2.dnn.blobFromImage(
            img_cv2, 1.0 / 255, (self.inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)
        # 向前通过网络
        self.pose_net.setInput(inpBlob)
        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        output = self.pose_net.forward()

        H = output.shape[2]
        W = output.shape[3]
        #print("形状：")
        #print(output.shape)

        # vis heatmaps
        #self.vis_bone_heatmaps(img_cv2, output)

        #
        points = []
        for idx in range(self.bone_num_points):
            # 把输出的大小调整到与输入一样
            probMap = output[0, idx, :, :]  # confidence map.

            # 提取关键点区域的局部最大值
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        # print(points)
        return points

    def vis_bone_pose(self, img_cv2, points):
        """显示标注骨骼点后的图像

        :param 图像，COCO检测关键点坐标
        :return 骨骼连线图、关键点图、骨骼连线黑背景图
        """
        #img_old = cv2.imread(imgfile)
        img_cv2_copy = np.copy(img_cv2)
        black_np = np.ones(img_cv2.shape, np.uint8)  # 创建黑色幕布
        for idx in range(len(points)):
            if points[idx]:
                cv2.circle(
                    black_np, points[idx], 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(
                    img_cv2_copy, points[idx], 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(img_cv2_copy, "{}".format(
                    idx), points[idx], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        """
        h = int(distance(points[4], points[3]))  # 小臂周长
        if points[4]:
            x_center = points[4][0]
            y_center = points[4][1]
            cv2.rectangle(img_cv2_copy, (x_center-h, y_center-h),
                          (x_center+h, y_center+h), (255, 0, 0), 2)  # 框
            cv2.circle(img_cv2_copy, (x_center, y_center), 1,
                       (100, 100, 0), thickness=-1, lineType=cv2.FILLED)  # 坐标点
            cv2.putText(img_cv2_copy, "%d,%d" % (x_center, y_center), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 100, 0), 2, lineType=cv2.LINE_AA)  # 右手首
        if points[7]:
            x_center = points[7][0]
            y_center = points[7][1]
            cv2.rectangle(img_cv2_copy, (x_center-h, y_center-h),
                          (x_center+h, y_center+h), (255, 0, 0), 1)
            cv2.putText(img_cv2_copy, "%d,%d" % (x_center, y_center), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 100, 0), 2, lineType=cv2.LINE_AA)  # 左手首
            cv2.circle(img_cv2_copy, (x_center-h, y_center-h), 3,
                       (225, 225, 255), thickness=-1, lineType=cv2.FILLED)  # 对角点
            cv2.putText(img_cv2_copy, "{}".format(x_center-h), (x_center-h, y_center-h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 100, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(img_cv2_copy, (x_center+h, y_center+h), 3,
                       (225, 225, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img_cv2_copy, "{}".format(x_center+h), (x_center+h, y_center+h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 100, 0), 2, lineType=cv2.LINE_AA)  # 对角点
        """
        # 骨骼连线
        for pair in self.bone_point_pairs:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(black_np, points[partA],
                         points[partB], (0, 255, 255), 2)
                cv2.line(img_cv2, points[partA],
                         points[partB], (0, 255, 255), 2)
                cv2.circle(
                    img_cv2, points[partA], 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # plt.figure(figsize=[10, 10])
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()

        return img_cv2,img_cv2_copy,black_np # 连线图，点图

    def vis_bone_heatmaps(self, img_cv2, net_outputs):
        """显示骨骼关键点热力图

        :param 图像路径，神经网络
        """
        #img_cv2 = cv2.imread(imgfile)
        plt.figure(figsize=[10, 10])
        for pdx in range(self.bone_num_points):
            probMap = net_outputs[0, pdx, :, :]  # 全部heatmap都初始化为0
            probMap = cv2.resize(probMap, (img_cv2.shape[1], img_cv2.shape[0]))
            plt.subplot(5, 5, pdx+1)
            plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))  # background
            plt.imshow(probMap, alpha=0.6)
            plt.colorbar()
            plt.axis("off")
        plt.show()