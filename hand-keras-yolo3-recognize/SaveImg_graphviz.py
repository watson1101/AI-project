#!/usr/bin/env python
import time
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter

from pose_hand import getImgInfo #引入外部函数和类
from pose.coco import general_coco_model
from yolo import YOLO
#也可以直接写在这里def getData():...

def main():
    config = Config()
    # 关系图中包括(include)哪些函数名。
    #如果是某一类的函数，例如类gobang，则可以直接写'gobang.*'，表示以gobang.开头的所有函数。（利用正则表达式）。
    config.trace_filter = GlobbingFilter(include=[
        'main',
        'pycallgraph.*',
        '*.secret_function',
    ])
    graphviz = GraphvizOutput()
    graphviz.output_file = 'basic.png'#图片名称

    with PyCallGraph(output=graphviz):
        # coco
        modelpath = "model/"
        start = time.time()
        pose_model = general_coco_model(modelpath)  # 1.加载模型
        print("[INFO]Pose Model loads time: ", time.time() - start)
        # yolo
        start = time.time()
        _yolo = YOLO() # 1.加载模型
        print("[INFO]yolo Model loads time: ", time.time() - start)

        img_path = 'D:/myworkspace/dataset/My_test/img/a_img/airplane_30.jpg'

        getImgInfo(img_path,pose_model,_yolo)



if __name__ == '__main__':
    main()
