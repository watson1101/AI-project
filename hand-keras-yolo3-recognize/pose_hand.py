import time
import numpy as np
from pose.coco import general_coco_model
from pose.hand import general_hand_model
from pose.data_process import getBoneInformation, getHandsInformation,getPoseAndYoloInfo
from pose.hand_fD import hand_fourierDesciptor
from yolo import YOLO
from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



def getImageInfo(img_file, modelpath):
    """获得所有信息

    :param 图片,模型路径
    :return list单图的信息
    """

    print("[INFO]Pose estimation.")
    start = time.time()
    pose_model = general_coco_model(modelpath)  # 1.加载模型

    print("[INFO]Model loads time: ", time.time() - start)

    # 骨骼
    start = time.time()
    img = cv2.imread(img_file)
    bone_points = pose_model.getBoneKeypoints(img)  # 2.骨骼关键点
    print("[INFO]COCO18_Model predicts time: ", time.time() - start)
    lineimage,dotimage,black_np = pose_model.vis_bone_pose(img, bone_points)  # 骨骼连线图、标记图显示cv2格式

    list1 = getBoneInformation(bone_points)  # 3.骨骼特征
    #print("[INFO]Model Bone Information[25]: ", list1)

    # yolo
    _yolo = YOLO()
    # cv2图片转PIL
    image = Image.open(img_file)
    lineimage = Image.fromarray(cv2.cvtColor(lineimage,cv2.COLOR_BGR2RGB))
    black_np = Image.fromarray(cv2.cvtColor(black_np,cv2.COLOR_BGR2RGB))
    r_image,labelinfo,hand_ROI_PIL = _yolo.detect_image(image,black_np) # 原图，lineimage线图,黑幕图
    # plt.figure(figsize=[5, 5])
    # plt.subplot(1, 2, 1)
    # plt.imshow(r_image)
    # plt.xlabel(u'线图', fontsize=20)
    # plt.axis("off")
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(dotimage, cv2.COLOR_BGR2RGB))
    # plt.xlabel(u'点图', fontsize=20)
    # plt.axis("off")
    # plt.show()

    # # 手势
    # print("[INFO]Hands estimation.by handpose")
    # start = time.time()
    # hand_model = general_hand_model(modelpath)  # 1.加载模型

    # hand_fd = hand_fourierDesciptor()

    # for i,handimg in enumerate(hand_ROI_PIL):
    #     img = cv2.cvtColor(np.asarray(handimg),cv2.COLOR_RGB2BGR)

    #     # hand 模型
    #     onehandpoints = hand_model.getOneHandKeypoints(img)
    #     hand_model.vis_hand_pose(img, onehandpoints)# 显示

    #     # hand_FD 描述子
    #     res1 = hand_fd.skinMask(img)  # 进行肤色检测
    #     ret1, fourier = hand_fd.fourierDesciptor(res1)  # 傅里叶描述子获取轮廓点

    info = []
    for i in range(len(list1)):
        info.append(list1[i])
    for j in range(len(labelinfo)):
        info.append(labelinfo[j])
    print(labelinfo)
    return r_image,info

def getVideo_Info(modelpath,video_path, output_path=""):
    from cv2 import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))  # 获取原始视频的信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False  # 如果设置了视频保存路径，则保存视频
    if isOutput:
        print("!!! TYPE:", type(output_path), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC,
                              video_fps, video_size)  # 根据原视频设置 保存视频的路径、大小、帧数
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    yolo = YOLO()
    pose_model = general_coco_model(modelpath)  # 1.加载模型
    while True:
        return_value, frame = cap.read()

        if return_value:
            # 骨骼
            bone_points = pose_model.getBoneKeypoints(frame)  # 2.骨骼关键点
            lineimage,dotimage = pose_model.vis_bone_pose(frame, bone_points)  # 骨骼连线图、标记图显示
            # list1 = getBoneInformation(bone_points)  # 3.骨骼特征

            temp,labelinfo = yolo.detect_image(Image.fromarray(frame),Image.fromarray(lineimage))  # 检测PIL格式
            result = np.asarray(temp) # 画图到全部图上
            curr_time = time.time()
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
        else:
            print("Frame is end!")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def getImgInfo(img,pose_model,_yolo,save_path=""):
    # 读取图像
    #image = cv2.imread(img_path)

    # 图像像素大小一致,视频保存尺寸有所冲突，这里弃用
    #img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    """ ################关键点坐标检测##################"""
    # pose骨骼
    start = time.time()
    bone_points = pose_model.getBoneKeypoints(img)  # 1.骨骼关键点
    #list1 = getBoneInformation(bone_points)  # 骨骼特征

    yololabel = _yolo.getyoloPoints(Image.fromarray(img)) # 2.yolo关键点
    print("[INFO]Model predicts time: ", time.time() - start)

    # print(bone_points)
    # print(yololabel)
    """ ################关键点距离角度特征信息##################"""
    list1=0
    list1 = getPoseAndYoloInfo(bone_points,yololabel)  # 全部特征
    # print(list1)

    """ ################绘图##################"""
    lineimage,dotimage,black_np = pose_model.vis_bone_pose(img, bone_points)  # pose绘图
    lineimage,dotimage,black_np = _yolo.vis_hand_pose(lineimage, dotimage, black_np, bone_points, yololabel) # yolo绘图
    # lineimage = Image.fromarray(cv2.cvtColor(lineimage,cv2.COLOR_BGR2RGB))# yolo绘图
    # black_np = Image.fromarray(cv2.cvtColor(black_np,cv2.COLOR_BGR2RGB))
    # line_image,yololabel,hand_ROI_PIL = _yolo.detect_image(image,black_np)

    isSave = True if save_path != "" else False  # 如果设置了视频保存路径，则保存视频
    if isSave:
        #lineimage.save(save_path)
        cv2.imwrite(save_path,lineimage,[int(cv2.IMWRITE_JPEG_QUALITY),70])
        #cv2.imwrite('1.png', img, [int(cv2.IMWRITE_JPEG_QUALITY),95])#图像的质量，用0 - 100的整数表示，默认95;对于png ,第三个参数表示的是压缩级别。默认为3.
        #cv2.imwrite('1.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])#从0到9 压缩级别越高图像越小

    # plt.figure(figsize=[5, 5])
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(lineimage, cv2.COLOR_BGR2RGB))
    # plt.xlabel(u'线图', fontsize=20)
    # plt.axis("off")
    # plt.subplot(1, 3, 2)
    # plt.imshow(cv2.cvtColor(dotimage, cv2.COLOR_BGR2RGB))
    # plt.xlabel(u'点图', fontsize=20)
    # plt.axis("off")
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(black_np, cv2.COLOR_BGR2RGB))
    # plt.xlabel(u'关键点线图', fontsize=20)
    # plt.axis("off")
    # plt.show()

    return list1, lineimage

def getVideoInfo(video_path,pose_model,yolo,output_path=""):
    from cv2 import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) # 获取原始视频的信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False  # 如果设置了视频保存路径，则保存视频
    if isOutput:
        print("[INFO] video TYPE:", type(output_path), type(fourcc), type(fps), type(size))
        out = cv2.VideoWriter(output_path, fourcc, fps, size)  # 根据原视频设置 保存视频的路径、大小、帧数
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    while True:
        ret, frame = cap.read()

        if ret:
            info, lineimage = getImgInfo(frame, pose_model, yolo) # 每一帧检测
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(lineimage, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.putText(lineimage, "q-'quit'", org=(3, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 255, 0), thickness=2)  # 标注字体
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", lineimage)
            if isOutput:
                out.write(lineimage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # coco
    modelpath = "model/"
    start = time.time()
    pose_model = general_coco_model(modelpath)  # 1.加载模型
    print("[INFO]Pose Model loads time: ", time.time() - start)
    # yolo
    start = time.time()
    _yolo = YOLO() # 1.加载模型
    print("[INFO]yolo Model loads time: ", time.time() - start)

    img_path = 'docs/classmates_5_2.png'

    img_path = 'docs/wangyu_hand_img/brave_92.jpg'
    image = cv2.imread(img_path)    
    #getImgInfo(image,pose_model,_yolo,'docs/brave_92_lineimg.jpg')
    #getVideoInfo("docs/sun.mp4",pose_model,_yolo,"docs/sun_detect.mp4")
    getVideoInfo("D:/myworkspace/dataset/My_test/video/wy2/you.mp4",pose_model,_yolo,"D:/myworkspace/dataset/My_test/you_detect.mp4")