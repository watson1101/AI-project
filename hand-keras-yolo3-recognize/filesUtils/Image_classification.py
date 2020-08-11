import os
import shutil
import re

path2 = 'D:/myworkspace/dataset/My_test/dataset/hand_classification/'

list = []
with open("D:/myworkspace/dataset/My_test/dataset/coco.txt", "r") as f:    #打开文件
    for line in f.readlines():   #读取文件
        list.append(line.strip())
        dir_name = os.path.join(path2, line.strip())
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
print(list)

path = os.walk(path2) # walk可以移动子目录
for root, dirs, files in path:
    for f in files:
        if ".png" in f:
            p = re.findall("(.*)_.*_.*", f)
            shutil.move(os.path.join(root, f), os.path.join(path2+p[0],f))  # move移动文件,copy复制文件
    