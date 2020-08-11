import os
from os import path

wdr = path.normpath(r'D:/myworkspace/dataset/img/b_img')
videoList = os.listdir(wdr)
#获取文件夹下所有文件列表
"""
压缩：
ffmpeg -i 1.mp4 -b:v 2000k 1_ffmpeg.mp4
去掉声音：
ffmpeg -i 1.mp4 -vcodec copy -an 2.mp4  

截图：
ffmpegCmd = 'ffmpeg -i {} -strict -2 -vf crop=1400:780:50:50 {}_conv.mp4 '
"""

#ffmpegCmd = 'ffmpeg -i {} -b:v 2000k {}_conv.mp4 '
ffmpegCmd = 'ffmpeg -i {} -vcodec copy -an {}_conv.mp4' 



#设置ffmpeg命令模板
cmd = f'cd "{wdr}"\n{path.splitdrive(wdr)[0]}\npause\n'
#第1步，进入目标文件夹

def comprehensionCmd(e):
    #手写一个小函数方便后面更新命令
    root,ext = path.splitext(e)
    return ffmpegCmd.format(e,root)

videoList = [comprehensionCmd(e) for e in videoList if not('conv' in e)]
#第3和第4步

cmd += '\n'.join(videoList)
# 将各个ffmpeg命令一行放一个

cmd += '\npause'


output = open('videoConv.bat','w')
output.write(cmd)
output.close()
#命令写入文件