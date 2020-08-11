import os
import shutil
import re

path = os.walk("D:/myworkspace/dataset/img/b_img") # walk可以移动子目录
for root, dirs, files in path:
    for f in files:
        if 'conv' in f:
            print(f)
            shutil.copy(os.path.join(root, f), os.path.join(
                'D:/myworkspace/dataset/img/compress', f))  # move移动文件,copy复制文件
# 重命名
folder_name = 'D:/myworkspace/dataset/img/compress'
for i, name in enumerate(os.listdir(folder_name)):
    e = re.findall("(.*)_.*", name)
    os.rename(os.path.join(folder_name, name),os.path.join(folder_name, str(e[0])+'.mp4'))