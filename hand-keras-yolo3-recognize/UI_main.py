# -*- coding: utf-8 -*-

import wx
import wx.xrc
import os
import time
from cv2 import cv2
import _thread
from sklearn.externals import joblib

from pose_hand import getImgInfo
from pose.coco import general_coco_model
from pose.hand import general_hand_model
from yolo import YOLO

from getKeyFrame import rel_change,smooth
from getKeyFrame import Frame
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema # 极值点
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

###########################################################################
# Class MyFrame1
###########################################################################

root_path = 'D:/myworkspace/JupyterNotebook/hand-keras-yolo3-recognize/'
#COVER = 'docs/ui/camera.png'
DEMO = 'docs/ui/demo.jpg'
ICO = 'docs/ui/favicon.ico'
App_title = 'Sign language detection system by cungudafa'

class MyFrame1 (wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=App_title, pos=wx.DefaultPosition, size=wx.Size(
            913, 641), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_INACTIVECAPTION))

        bSizer1 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer3 = wx.BoxSizer(wx.VERTICAL)

        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox(
            self, wx.ID_ANY, u"原始图像"), wx.VERTICAL)

        self.orgin_img = wx.StaticBitmap(sbSizer1.GetStaticBox(
        ), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer1.Add(self.orgin_img, 1, wx.ALL | wx.EXPAND, 5)

        bSizer3.Add(sbSizer1, 1, wx.EXPAND, 5)

        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(
            self, wx.ID_ANY, u"人姿和手部检测"), wx.VERTICAL)

        self.result_img = wx.StaticBitmap(sbSizer2.GetStaticBox(
        ), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer2.Add(self.result_img, 1, wx.ALL | wx.EXPAND, 5)

        bSizer3.Add(sbSizer2, 1, wx.EXPAND, 5)

        bSizer2.Add(bSizer3, 1, wx.EXPAND, 5)

        bSizer4 = wx.BoxSizer(wx.VERTICAL)

        sbSizer3 = wx.StaticBoxSizer(wx.StaticBox(
            self, wx.ID_ANY, u"运行日志"), wx.VERTICAL)

        self.tips = wx.TextCtrl(sbSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString,
                                wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE | wx.TE_READONLY)
        sbSizer3.Add(self.tips, 1, wx.ALL | wx.EXPAND, 5)

        bSizer4.Add(sbSizer3, 1, wx.EXPAND, 5)

        sbSizer4 = wx.StaticBoxSizer(wx.StaticBox(
            self, wx.ID_ANY, u"检测结果"), wx.VERTICAL)

        self.result = wx.TextCtrl(sbSizer4.GetStaticBox(), wx.ID_ANY, wx.EmptyString,
                                  wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE | wx.TE_READONLY)
        sbSizer4.Add(self.result, 1, wx.ALL | wx.EXPAND, 5)

        bSizer4.Add(sbSizer4, 1, wx.EXPAND, 5)

        bSizer2.Add(bSizer4, 1, wx.EXPAND, 5)

        bSizer5 = wx.BoxSizer(wx.VERTICAL)

        sbSizer5 = wx.StaticBoxSizer(wx.StaticBox(
            self, wx.ID_ANY, u"控制台"), wx.VERTICAL)

        sbSizer6 = wx.StaticBoxSizer(wx.StaticBox(
            sbSizer5.GetStaticBox(), wx.ID_ANY, u"选择视频或文件"), wx.HORIZONTAL)

        self.m_button7 = wx.Button(sbSizer6.GetStaticBox(
        ), wx.ID_ANY, u"选择图片", wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer6.Add(self.m_button7, 1, wx.ALL | wx.EXPAND, 5)

        self.m_button8 = wx.Button(sbSizer6.GetStaticBox(
        ), wx.ID_ANY, u"选择视频", wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer6.Add(self.m_button8, 1, wx.ALL | wx.EXPAND, 5)

        sbSizer5.Add(sbSizer6, 0, wx.EXPAND, 5)

        sbSizer7 = wx.StaticBoxSizer(wx.StaticBox(
            sbSizer5.GetStaticBox(), wx.ID_ANY, u"特征提取"), wx.VERTICAL)

        self.m_button1 = wx.Button(sbSizer7.GetStaticBox(
        ), wx.ID_ANY, u"特征提取", wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer7.Add(self.m_button1, 0, wx.ALL | wx.EXPAND, 5)

        sbSizer5.Add(sbSizer7, 0, wx.EXPAND, 5)

        sbSizer8 = wx.StaticBoxSizer(wx.StaticBox(
            sbSizer5.GetStaticBox(), wx.ID_ANY, u"手语预测"), wx.VERTICAL)

        self.m_button5 = wx.Button(sbSizer8.GetStaticBox(
        ), wx.ID_ANY, u"选择分类模型", wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer8.Add(self.m_button5, 0, wx.ALL | wx.EXPAND, 5)

        self.m_button6 = wx.Button(sbSizer8.GetStaticBox(
        ), wx.ID_ANY, u"预测", wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer8.Add(self.m_button6, 0, wx.ALL | wx.EXPAND, 5)

        sbSizer5.Add(sbSizer8, 0, wx.EXPAND, 5)

        sbSizer9 = wx.StaticBoxSizer(wx.StaticBox(
            sbSizer5.GetStaticBox(), wx.ID_ANY, u"示例图"), wx.VERTICAL)

        self.black_img = wx.StaticBitmap(sbSizer9.GetStaticBox(
        ), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0)
        sbSizer9.Add(self.black_img, 1, wx.ALL | wx.EXPAND, 5)

        sbSizer5.Add(sbSizer9, 1, wx.EXPAND, 5)

        bSizer5.Add(sbSizer5, 1, wx.EXPAND, 5)

        bSizer2.Add(bSizer5, 1, wx.EXPAND, 5)

        bSizer1.Add(bSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()
        self.m_menubar1 = wx.MenuBar(0)
        self.m_menubar1.SetForegroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        self.m_menubar1.SetBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

        self.m_menu1 = wx.Menu()
        self.m_menuItem1 = wx.MenuItem(
            self.m_menu1, wx.ID_ANY, u"图片", wx.EmptyString, wx.ITEM_NORMAL)
        self.m_menu1.Append(self.m_menuItem1)

        self.m_menuItem2 = wx.MenuItem(
            self.m_menu1, wx.ID_ANY, u"视频", wx.EmptyString, wx.ITEM_NORMAL)
        self.m_menu1.Append(self.m_menuItem2)

        self.m_menubar1.Append(self.m_menu1, u"检测")

        self.m_menu2 = wx.Menu()
        self.m_menubar1.Append(self.m_menu2, u"采集")

        self.m_menu3 = wx.Menu()
        self.m_menubar1.Append(self.m_menu3, u"关于")

        self.SetMenuBar(self.m_menubar1)

        self.m_statusBar2 = self.CreateStatusBar(1, wx.STB_SIZEGRIP, wx.ID_ANY)
        self.m_statusBar2.SetForegroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))
        self.m_statusBar2.SetBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_ACTIVECAPTION))

        self.Centre(wx.BOTH)

        # Connect Events
        self.m_button7.Bind(wx.EVT_BUTTON, self.img_btn)
        self.m_button8.Bind(wx.EVT_BUTTON, self.video_btn)
        self.m_button1.Bind(wx.EVT_BUTTON, self.feature_btn)
        self.m_button5.Bind(wx.EVT_BUTTON, self.chosemodel_btn)
        self.m_button6.Bind(wx.EVT_BUTTON, self.predict_btn)

        """界面"""
        self.image_demo = wx.Image(DEMO, wx.BITMAP_TYPE_ANY).Scale(500,400)
        self.bmp = wx.StaticBitmap(
            self.black_img, -1, wx.Bitmap(self.image_demo))

        # 设置窗口标题的图标
        self.icon = wx.Icon(ICO, wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        print("wxpython界面初始化加载完成！")

        """参数"""
        #
        self.VIDEO_STREAM = False
        self.IMAGE_STREAM = False
        self.orgin_img_show = root_path+'docs/images/brave_40.jpg' # 默认预测图片
        self.model_path = root_path+'model/' # 训练的深度模型
        self._loadmodel() # 加载OpenPose和yolo模型

        self.beyes_model = root_path+'model/train_model.m' # 默认贝叶斯模型
        self.XX_test = [] # 测试集

    def __del__(self):
        pass

    # Virtual event handlers, overide them in your derived class
    def img_btn(self, event):
        self.IMAGE_STREAM = True
        dialog = wx.FileDialog(self, u"选择图片检测", os.getcwd(
        ), '', wildcard="(*.jpg)|*.jpg|(*.png)|*.png", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            # 如果确定了选择的文件夹，将文件夹路径写到tips控件
            self.tips.SetValue(u"文件路径:"+dialog.GetPath()+"\n")
            self.orgin_img_show =  cv2.imread(str(dialog.GetPath())) # 更新全局变量路径
            dialog.Destroy
        # cv2转wxpython
        self.orgin_img_show = cv2.resize(self.orgin_img_show,(600,500),)
        height, width = self.orgin_img_show.shape[:2]
        image1 = cv2.cvtColor(self.orgin_img_show, cv2.COLOR_BGR2RGB)
        pic = wx.Bitmap.FromBuffer(width, height, image1)
        # 显示图片在panel上：
        self.orgin_img.SetBitmap(pic)

    def video_btn(self, event):
        self.VIDEO_STREAM = True
        # 选择文件夹对话框窗口
        dialog = wx.FileDialog(self, u"选择视频检测", os.getcwd(
        ), '', wildcard="(*.mp4)|*.mp4", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            # 如果确定了选择的文件夹，将文件夹路径写到tips控件
            self.tips.SetValue(u"文件路径:"+dialog.GetPath()+"\n")
            self.video_path = str(dialog.GetPath())# 更新全局变量路径
            dialog.Destroy
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._getEffectiveFrame, (event,))

    def feature_btn(self, event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_hand, (event,))

    def chosemodel_btn(self, event):
        event.Skip()
        # beye模型
        dialog = wx.FileDialog(self, u"选择分类器", os.getcwd()+"/model/", '',
         wildcard="(*.pkl)|*.pkl|(*.m)|*.m", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            # 如果确定了选择的文件夹，将文件夹路径写到tips控件
            self.tips.AppendText(u"分类器模型路径:"+dialog.GetPath()+"\n")
            self.beyes_model = str(dialog.GetPath()) # 更新全局变量路径
            dialog.Destroy

    def predict_btn(self, event):
        start = time.time()
        
        self.clf = joblib.load(self.beyes_model) # 加载分类器模型
        predictions_labels = self.clf.predict(self.XX_test) # 预测
        self.result.AppendText(u"预测结果:\n"+str(predictions_labels)+"\n")
        self.result.AppendText(u"预测耗时:{:.2f} s".format(time.time() - start)+"\n")

    def _loadmodel(self):
        start = time.time()
        self.pose_model = general_coco_model(self.model_path)  # coco加载模型
        self._yolo = YOLO()  # 1.加载模型yolo
        self.tips.SetValue(u"模型加载耗时:{:.2f} s".format(time.time() - start)+"\n")

    def _learning_hand(self, event):
        start = time.time()
        info, lineimage = getImgInfo(self.orgin_img_show, self.pose_model, self._yolo, '')

        self.XX_test.append(info)
        self.tips.AppendText(u"特征提取耗时:{:.2f} s".format(time.time() - start)+"\n")
        lineimage = cv2.resize(lineimage,(600,500),)
        height, width = lineimage.shape[:2]
        image1 = cv2.cvtColor(lineimage, cv2.COLOR_BGR2RGB)# opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
        pic = wx.Bitmap.FromBuffer(width, height, image1)
        # 显示图片在panel上：
        self.result_img.SetBitmap(pic)
        self.tips.AppendText(u"数字特征:"+str(info)+"\n")

    def _getEffectiveFrame(self,event):
        (filepath, tempfilename) = os.path.split(self.video_path)#分离路径和文件名
        (filename, extension) = os.path.splitext(tempfilename)#区分文件的名字和后缀
        #Setting fixed threshold criteria设置固定阈值标准
        USE_THRESH = False
        #fixed threshold value固定阈值
        THRESH = 0.6
        #Setting fixed threshold criteria设置固定阈值标准
        USE_TOP_ORDER = False
        #Setting local maxima criteria设置局部最大值标准
        USE_LOCAL_MAXIMA = True
        #Number of top sorted frames排名最高的帧数
        NUM_TOP_FRAMES = 50
        #smoothing window size平滑窗口大小
        len_window = int(50)
        #self.tips.AppendText(u"视频路径:"+self.video_path+"\n")
        # load video and compute diff between frames加载视频并计算帧之间的差异
        cap = cv2.VideoCapture(self.video_path) 
        curr_frame = None
        prev_frame = None 
        frame_diffs = []
        frames = []
        success, frame = cap.read()
        i = 0 
        while(success):
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            curr_frame = luv
            if curr_frame is not None and prev_frame is not None:
                #logic here
                diff = cv2.absdiff(curr_frame, prev_frame)#获取差分图
                diff_sum = np.sum(diff)
                diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])#平均帧
                frame_diffs.append(diff_sum_mean)
                frame = Frame(i, diff_sum_mean)
                frames.append(frame)
            prev_frame = curr_frame
            i = i + 1
            success, frame = cap.read()   
        cap.release()
        
        # compute keyframe
        keyframe_id_set = set()
        if USE_TOP_ORDER:
            # sort the list in descending order以降序对列表进行排序
            frames.sort(key=operator.attrgetter("diff"), reverse=True)# 排序operator.attrgetter
            for keyframe in frames[:NUM_TOP_FRAMES]:
                keyframe_id_set.add(keyframe.id) 
        if USE_THRESH:
            print("Using Threshold")#使用阈值
            for i in range(1, len(frames)):
                if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                    keyframe_id_set.add(frames[i].id)   
        if USE_LOCAL_MAXIMA:
            print("Using Local Maxima")#使用局部极大值
            diff_array = np.array(frame_diffs)
            sm_diff_array = smooth(diff_array, len_window)#平滑
            frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]#找极值
            for i in frame_indexes:
                keyframe_id_set.add(frames[i - 1].id)# 记录极值帧数
                
            # self.keyFrame_img = plt.figure(figsize=(40, 20))
            # plt.locator_params("x", nbins = 100)
            # # stem 绘制离散函数，polt是连续函数
            # plt.stem(sm_diff_array,linefmt='-',markerfmt='o',basefmt='--',label='sm_diff_array')
            #plt.savefig(dirfile + filename+'_plot.png')
        
        # save all keyframes as image将所有关键帧另存为图像
        cap = cv2.VideoCapture(str(self.video_path))
        curr_frame = None
        keyframes = []
        success, frame = cap.read()
        idx = 0
        while(success):
            if idx in keyframe_id_set:
                name = filename+'_' + str(idx) + ".jpg"
                #cv2.imwrite(dirfile + name, frame)
                self.tips.AppendText(u"极值帧数:"+ name +"\n")
                self.orgin_img = frame
                _thread.start_new_thread(self._learning_hand, (event,)) # 关键点append
                keyframe_id_set.remove(idx)
            idx = idx + 1
            success, frame = cap.read()
        cap.release()
    
    

def main():
    app = wx.App(False)
    frame = MyFrame1(None)
    frame.Show(True)
    # start the applications
    app.MainLoop()


if __name__ == '__main__':
    main()