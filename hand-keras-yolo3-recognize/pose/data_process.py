import math

def distance(A, B):
    """距离辅助函数

    :param 两个坐标A(x1,y1)B(x2,y2)
    :return 距离d=AB的距离
    """
    if A is None or B is None:
        return 0
    else:
        return math.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)


def myAngle(A, B, C):
    """角度辅助函数

    :param 三个坐标A(x1,y1)B(x2,y2)C(x3,y3)
    :return 角B的余弦值（转换为角度）
    """
    if A is None or B is None or C is None:
        return 0
    else:
        a =   distance(B, C)
        b =   distance(A, C)
        c =   distance(A, B)
        if 2*a*c != 0:
            return (a**2/+c**2-b**2)/(2*a*c)  # math.degrees计算出cos弧度，转换为角度
        return 0

"""手20特征点：右手手指尖距离、左手、右手手指角度、左手"""
# 左右手分别手指间距（10个位置）
def handDistance(rkeyPoint, lkeyPoint):
    """距离辅助函数
    :param keyPoint:
    :return:list
    :distance:
    """
    if rkeyPoint[0] is None or lkeyPoint[0] is None:
        print("未识别到Wrist参考关键点")
    distance0 =   distance(rkeyPoint[0], rkeyPoint[4])  # Thumb拇指
    distance1 =   distance(rkeyPoint[0], rkeyPoint[8])  # Index食指
    distance2 =   distance(rkeyPoint[0], rkeyPoint[12])  # Middle中指
    distance3 =   distance(rkeyPoint[0], rkeyPoint[16])  # Ring无名指
    distance4 =   distance(rkeyPoint[0], rkeyPoint[20])  # Little小指
    #
    distance5 =   distance(lkeyPoint[0], lkeyPoint[4])  # Thumb拇指
    distance6 =   distance(lkeyPoint[0], lkeyPoint[8])  # Index食指
    distance7 =   distance(lkeyPoint[0], lkeyPoint[12])  # Middle中指
    distance8 =   distance(lkeyPoint[0], lkeyPoint[16])  # Ring无名指
    distance9 =   distance(lkeyPoint[0], lkeyPoint[20])  # Little小指

    return [distance0, distance1, distance2, distance3, distance4, distance5, distance6, distance7, distance8, distance9]

# 左右手分别的手指角度（10）
def handpointAngle(rkeyPoint, lkeyPoint):
    """角度辅助函数

    :param keyPoint:
    :return:list
    :角度: 
    """
    angle0 =   myAngle(rkeyPoint[0], rkeyPoint[9], rkeyPoint[4])
    angle1 =   myAngle(rkeyPoint[0], rkeyPoint[9], rkeyPoint[8])
    angle2 =   myAngle(rkeyPoint[0], rkeyPoint[9], rkeyPoint[12])
    angle3 =   myAngle(rkeyPoint[0], rkeyPoint[9], rkeyPoint[16])
    angle4 =   myAngle(rkeyPoint[0], rkeyPoint[9], rkeyPoint[20])
    #
    angle5 =   myAngle(lkeyPoint[0], lkeyPoint[9], lkeyPoint[4])
    angle6 =   myAngle(lkeyPoint[0], lkeyPoint[9], lkeyPoint[8])
    angle7 =   myAngle(lkeyPoint[0], lkeyPoint[9], lkeyPoint[12])
    angle8 =   myAngle(lkeyPoint[0], lkeyPoint[9], lkeyPoint[16])
    angle9 =   myAngle(lkeyPoint[0], lkeyPoint[9], lkeyPoint[20])
    return [angle0, angle1, angle2, angle3, angle4,angle5, angle6, angle7, angle8, angle9]

def getHandsInformation(rpoints, lpoints): # 左右手距离、角度共20特征点
    """将右手和左手（各距离5和角度5个特征）信息汇集

    :param 左右手关键点
    :return 左右手距离、角度共20特征点
    """
    Information = []
    # 距离
    DistanceList = handDistance(rpoints,lpoints)  # 左右手汇总
    for i in range(len(DistanceList)):
        Information.append(DistanceList[i])
    # 角度
    AngleList = handpointAngle(rpoints,rpoints)
    for j in range(len(AngleList)):
        Information.append(AngleList[j])
    return Information

"""手20特征点：右手手指尖距离、左手、右手手指角度、左手"""
# 上半身身体距离信息（17）
def bonepointDistance(keyPoint):
    """距离辅助函数
    :param keyPoint:
    :return:list
    :distance:
    """
    r = distance(keyPoint[0], keyPoint[1])# 参考距离 脖子
    if r == 0: # 分母不为0
        r = 1
    d0 =   distance(keyPoint[4], keyPoint[8])  # 右手右腰
    d1 =   distance(keyPoint[7], keyPoint[11])  # 左手左腰
    d2 =   distance(keyPoint[2], keyPoint[4])  # 手肩
    d3 =   distance(keyPoint[5], keyPoint[7])
    d4 =   distance(keyPoint[0], keyPoint[4])  # 头手
    d5 =   distance(keyPoint[0], keyPoint[7])
    d6 =   distance(keyPoint[4], keyPoint[7])  # 两手
    d7 =   distance(keyPoint[4], keyPoint[16])  # 手耳
    d8 =   distance(keyPoint[7], keyPoint[17])
    d9 =   distance(keyPoint[4], keyPoint[14])  # 手眼
    d10 =   distance(keyPoint[7], keyPoint[15])
    d11 =   distance(keyPoint[4], keyPoint[1])  # 手脖
    d12 =   distance(keyPoint[7], keyPoint[1])
    d13 =   distance(keyPoint[4], keyPoint[5])  # 左手左臂
    d14 =   distance(keyPoint[4], keyPoint[6])  # 右手左肩
    d15 =   distance(keyPoint[7], keyPoint[2])  # 右手左肩
    d16 =   distance(keyPoint[7], keyPoint[3])  # 左手右臂

    return [d0/r*100, d1/r*100, d2/r*100, d3/r*100, d4/r*100, d5/r*100, d6/r*100, d7/r*100, d8/r*100,
            d9/r*100, d10/r*100, d11/r*100,d12/r*100, d13/r*100, d14/r*100, d15/r*100, d16/r*100]

# 上半身身体角度信息（8）
def bonepointAngle(keyPoint):
    """角度辅助函数

    :param keyPoint:
    :return:list
    :角度:
    """
    angle0 =   myAngle(keyPoint[2], keyPoint[3], keyPoint[4])  # 右手臂夹角
    angle1 =   myAngle(keyPoint[5], keyPoint[6], keyPoint[7])  # 左手臂夹角
    angle2 =   myAngle(keyPoint[3], keyPoint[2], keyPoint[1])  # 右肩夹角
    angle3 =   myAngle(keyPoint[6], keyPoint[5], keyPoint[1])
    angle4 =   myAngle(keyPoint[4], keyPoint[0], keyPoint[7])  # 头手头
    if keyPoint[8] is None or keyPoint[11] is None:
        angle5 = 0
    else:
        temp = ((keyPoint[8][0]+keyPoint[11][0])/2,
                (keyPoint[8][1]+keyPoint[11][1])/2)  # 两腰的中间值
        angle5 =   myAngle(keyPoint[4], temp, keyPoint[7])  # 手腰手
    angle6 =   myAngle(keyPoint[4], keyPoint[1], keyPoint[8])  # 右手脖腰
    angle7 =   myAngle(keyPoint[7], keyPoint[1], keyPoint[11])  # 右手脖腰

    return [angle0, angle1, angle2, angle3, angle4, angle5, angle6, angle7]

def getBoneInformation(bone_points): # 上半身距离和角度25个特征信息
    """将距离和角度25个特征信息汇集

    :param 骨骼关键点
    :return list 上半身距离和角度25个特征信息
    """
    Information = []
    DistanceList = bonepointDistance(bone_points)  # 3. 距离关键信息
    AngleList = bonepointAngle(bone_points)  # 4. 角度关键信息
    for i in range(len(DistanceList)):
        Information.append(DistanceList[i])
    for j in range(len(AngleList)):
        Information.append(AngleList[j])

    return Information


def yoloDistanceAndAngle(bone_points,yololabel):
    r = distance(bone_points[0], bone_points[1])# 参考距离 脖子
    if r == 0: # 分母不为0
        r = 1
    if yololabel[0]==0:
        return [0,0,0,0,0,0]
    if yololabel[0]==1: # 一只手
        distance0 =   distance(bone_points[1], yololabel[2])
        distance1 =   distance(bone_points[1], yololabel[3])
        angle0 =   myAngle(yololabel[2], bone_points[1], yololabel[3])
        return [distance0/r*100,distance1/r*100,angle0,0,0,0]
    if yololabel[0] > 1: # 2只手
        distance0 =   distance(bone_points[1], yololabel[2])
        distance1 =   distance(yololabel[2], yololabel[3])
        angle0 =   myAngle(yololabel[2], bone_points[1], yololabel[3])
        distance2 =   distance(bone_points[1], yololabel[5])
        distance3 =   distance(bone_points[1], yololabel[6])
        angle1 =   myAngle(yololabel[5], bone_points[1], yololabel[6])
        return [distance0/r*100,distance1/r*100,angle0,distance2/r*100,distance3/r*100,angle1]
    return [0,0,0,0,0,0]

def getPoseAndYoloInfo(bone_points,yololabel):
    Information = []
    DistanceList = bonepointDistance(bone_points)  # 3. 距离关键信息
    AngleList = bonepointAngle(bone_points)  # 4. 角度关键信息
    Information.extend(DistanceList)
    Information.extend(AngleList)
    YoloList = yoloDistanceAndAngle(bone_points,yololabel)# 4. 角度关键信息
    Information.extend(YoloList)
    return Information


