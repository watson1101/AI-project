import os
import re

folder_name = 'D:/myworkspace/dataset/My_test/small_img/23_img'

names=[]
for i, name in enumerate(os.listdir(folder_name)):
    names.append(name)
print(names)

folder_name = 'D:/myworkspace/dataset/My_test/img/01wy_img'
test_id=1
for i, name in enumerate(os.listdir(folder_name)):
    p=names[i]
    p = re.findall("(.*)_.*_.*", names[i])
    num = re.findall(".*_(.*)_.*", names[i])
    e = re.findall(".*_.*_(.*)", names[i])
    print(str(p[0]),str( int(num[0])-2),str( e[0]))
    os.rename(os.path.join(folder_name, name),os.path.join(folder_name,str(p[0])+"_"+str( int(num[0])-2)+"_"+str( e[0])))

    test_id=+1

print(test_id)

