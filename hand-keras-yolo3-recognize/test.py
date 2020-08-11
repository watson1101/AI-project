#-*- coding:utf-8 -*-
import os
import sys
import re
import numpy as np
import pandas as pd
#import openpyxl
import xlrd


def checkText(file_dir,queryText):
    wb = xlrd.open_workbook(file_dir) #打开excel表

    #通过索引获取  从0开始
    sheet = wb.sheet_by_index(1)
    #通过名字获取某个sheet页的值
    #sheet = wb.sheet_by_name('成果物)

    #初始化count
    count =0
    #获取行数
    nrows = sheet.nrows
    #获取总列数
    ncols = sheet.ncols

    qcols = 4 #检索列 注意：数组从0开始

    #print("The sum rows:%d" %nrows)
    #print("The sum cols:%d" %ncols)


    #获取列数
    for i in range(nrows):
        if re.search(queryText,sheet.cell_value(i,qcols)):
            count = count +1
    if count>0:
        print(file_dir)


#根据文件夹 截取文件名称
def getFileList(path ):
    allfileList = os.listdir(path)
    count = 0
    for file in allfileList:
        filepath = os.path.join(path,file)
        #只取以xlsx文件结尾的文件
        if re.search(r'.xlsx', filepath):
            #获取excel文件内容，并判断是否包含
            checkText(filepath,'感恩')

if __name__ == '__main__':
    path ="D:/myworkspace/dataset/"
    getFileList(path)