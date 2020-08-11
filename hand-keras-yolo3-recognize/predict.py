#!/usr/bin/env python
import os
from keras.layers import Input
from yolo import YOLO, detect_video
from pose_hand import getImageInfo,getVideoInfo

from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def main(yolo):

    #img_file = 'docs/wangyu_hand_img/job_32.jpg'#'docs/img/write_51.jpg'
    #print("[INFO]img_file",img_file)

    path= 'D:/myworkspace/dataset/My_test/small_img/67_img/'
    outfile = 'D:/myworkspace/dataset/My_test/hand_img/67_img/'#处理完的帧
    if not os.path.exists(outfile):  
        os.makedirs(outfile)
    files = [os.path.join(path, file1) for file1 in os.listdir(path)]
    # pose
    modelpath = "model/"
    for img_file in files:
        name = os.path.split(img_file)[1]
        if name[-4:] == ".png":
            r_image,info = getImageInfo(img_file,modelpath)
            print(info)
            r_image.save(outfile + name[:-4]+".jpg")# 保存
            print("成功保存",outfile,name[:-4]+".jpg")

    #r_image,info = getImageInfo(img_file,modelpath)
    #print("[INFO]Pose[25] and Hands[20]: ", info)

    video_name = "docs/write"
    #getVideoInfo(modelpath,video_name+".mp4", video_name+"_detect.mp4")


if __name__ == '__main__':
    _yolo = YOLO()

    main(_yolo)