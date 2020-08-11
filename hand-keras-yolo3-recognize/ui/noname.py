# -*- coding: utf-8 -*-

###########################################################################
# Python code generated with wxFormBuilder (version Jan 23 2018)
# http://www.wxformbuilder.org/
##
# PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
# Class MyFrame1
###########################################################################


class MyFrame1 (wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition, size=wx.Size(
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

    def __del__(self):
        pass

    # Virtual event handlers, overide them in your derived class
    def img_btn(self, event):
        event.Skip()

    def video_btn(self, event):
        event.Skip()

    def feature_btn(self, event):
        event.Skip()

    def chosemodel_btn(self, event):
        event.Skip()

    def predict_btn(self, event):
        event.Skip()
