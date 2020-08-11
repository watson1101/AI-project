# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from pose.coco import general_coco_model


class YOLO(object):
    _defaults = {
        "model_path": 'D:/myworkspace/JupyterNotebook/hand-keras-yolo3-recognize/model/yolov3/last1.h5',  # 模型
        "anchors_path": 'D:/myworkspace/JupyterNotebook/hand-keras-yolo3-recognize/model/yolov3/coco_anchors.txt',  # 先验框
        "classes_path": 'D:/myworkspace/JupyterNotebook/hand-keras-yolo3-recognize/model/yolov3/voc_classes.txt',  # 种类
        "score": 0.3,  # 框置信度阈值，小于阈值的框被删除，需要的框较多，则调低阈值，需要的框较少，则调高阈值
        "iou": 0.45,  # 同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.load_yolo_model()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        #print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(self.colors)
        np.random.seed(None)  # Reset seed to default.

    def compute_output(self, image_data, image_shape):
        # Generate output tensor targets for filtered bounding boxes.
        # self.input_image_shape = K.placeholder(shape=(2,))
        self.input_image_shape = tf.constant(image_shape)
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(
                self.yolo_model, gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model(image_data), self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def getyoloPoints(self, image):
        """yolo手部关键点检测

        :param 图像
        :return 信息[手的个数,结果,左上角坐标,右下角坐标,...]
        """
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))  # 原图转换成数组格式
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.compute_output(
            image_data, [image.size[1], image.size[0]])  # yolo检测结果
        if len(out_boxes) > 0:
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        Info = []  # 存放信息的列表
        Info.append(len(out_boxes))
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))#打印检测结果、左上角坐标和右下角坐标
            Info.append(format(score))
            Info.append((left, top))
            Info.append((right, bottom))

        end = timer()
        if len(out_boxes) > 0:
            print('[INFO]yolo_Model predicts time: {}'.format(end - start))
        return Info

    def vis_hand_pose(self, img_cv2, dotimg, black_np,bone_points, yololabel):
        from cv2 import cv2
        """显示yolo寻找到手势位置的图像

        :param 图像，yolo信息包含检测关键点坐标
        :return 手势框
        """
        #img_old = cv2.imread(imgfile)

        points = []
        if yololabel[0] == 1:  # 一只手
            points.append(yololabel[2])
            points.append(yololabel[3])
            # cvRectangle函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型 
            cv2.rectangle(img_cv2, yololabel[2],
                          yololabel[3], (255, 0, 0), 1, 4, 0 )  # 画yolo矩形
            cv2.rectangle(black_np, yololabel[2],
                          yololabel[3], (255, 0, 0), 1, 4, 0 )
            cv2.rectangle(dotimg, yololabel[2],
                          yololabel[3], (255, 0, 0), 1, 4, 0 )
        if yololabel[0] > 1:  # 2只手
            points.append(yololabel[2])
            points.append(yololabel[3])
            cv2.rectangle(img_cv2, yololabel[2], yololabel[3], (255, 0, 0), 1, 4, 0 )
            cv2.rectangle(black_np, yololabel[2],
                          yololabel[3], (255, 0, 0), 1, 4, 0 )
            cv2.rectangle(dotimg, yololabel[2],
                          yololabel[3], (255, 0, 0), 1, 4, 0 )
            points.append(yololabel[5])
            points.append(yololabel[6])
            cv2.rectangle(img_cv2, yololabel[5], yololabel[6], (255, 0, 0), 1, 4, 0 )
            cv2.rectangle(black_np, yololabel[5],
                          yololabel[6], (255, 0, 0), 1, 4, 0 )
            cv2.rectangle(dotimg, yololabel[5],
                          yololabel[6], (255, 0, 0), 1, 4, 0 )
        for idx,yolopoint in enumerate(points):
            if yolopoint:
                # 线
                cv2.line(black_np, yolopoint,
                         bone_points[1], (0, 255, 0), 2) # 手与骨骼点1的连线
                cv2.line(dotimg, yolopoint,
                         bone_points[1], (0, 255, 0), 2)
                cv2.line(img_cv2, yolopoint,
                         bone_points[1], (0, 255, 0), 2)
                # 点
                cv2.circle(
                    black_np, yolopoint, 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED) # yolo手点
                cv2.circle(
                    img_cv2, yolopoint, 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(
                    dotimg, yolopoint, 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
        # 骨骼点1
        cv2.circle(
                    img_cv2, bone_points[1], 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(
                    dotimg, bone_points[1], 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED) 
        cv2.circle(
                    black_np, bone_points[1], 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED) 
        return img_cv2, dotimg, black_np  # 连线图，点图,黑连线图

    def detect_image(self, image, lineimage):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))  # 原图转换成数组格式
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.compute_output(
            image_data, [image.size[1], image.size[0]])  # yolo检测结果
        if len(out_boxes) > 0:
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='docs/font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        Info = []  # 存放信息的列表
        hand_ROI_PIL = []  # 存放手部图片
        Info.append(len(out_boxes))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)  # 打印标签类型、相似度
            draw = ImageDraw.Draw(lineimage)  # 画在点线图上
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))#打印检测结果、左上角坐标和右下角坐标
            Info.append(format(score))
            Info.append((left, top))
            Info.append((right, bottom))
            # 从左上角开始 剪切图片
            img2 = image.crop((left, top, right, bottom))  # 原图中手的图片，处理
            # img2.save("docs/"+label+"_"+str(i)+".jpg")
            hand_ROI_PIL.append(img2)

            # 画图
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])  # 在点线图中框出手，显示
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])  # 写hand识别率的框
            draw.text(text_origin, label, fill=(
                0, 0, 0), font=font)  # 写上标签和识别率
            del draw

        end = timer()
        if len(out_boxes) > 0:
            print('[INFO]yolo_Model predicts time: {}'.format(end - start))
        return lineimage, Info, hand_ROI_PIL


def detect_video(yolo, video_path, output_path=""):
    from cv2 import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))  # 获取原始视频的信息
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False  # 如果设置了视频保存路径，则保存视频
    if isOutput:
        print("!!! TYPE:", type(output_path), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC,
                              video_fps, video_size)  # 根据原视频设置 保存视频的路径、大小、帧数
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        image = Image.fromarray(frame)
        lineimage, Info, hand_ROI_PIL = yolo.detect_image(image,image)  # 检测
        result = np.asarray(Image.fromarray(lineimage))  # 画图到全部图上
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.putText(result, "q-'quit'", org=(3, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 0), thickness=2)  # 标注字体
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
