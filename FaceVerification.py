# -*- coding: utf-8 -*-
from facepp import API
from facepp import File

dataBasePath = 'E:\\code\\Python\\FaceVerification\\data\\'
groupName = 'alwTest3'


'''
描述：是否需要训练
参数：[out]True,需要训练，False，不需要训练
'''
def isNeedTrain(facepp):
    flag = True
    groupDict = facepp.info.get_group_list()
    for group in groupDict['group']:
        if group['group_name'] == groupName:
            flag = False
            break
    if flag:
        facepp.group.create(group_name = groupName)
        return True
    return False

'''
描述：训练数据集
参数：[in]trainDataPath,str,训练数据集的路径
'''
def train(trainDataPath):
    dataSet = phraseDataList(trainDataPath)
    for data in dataSet:
        pic1Path = data['pic1']
        pic2Path = data['pic2']
        result1 = facepp.detection.detect(img=File(pic1Path), mode = 'oneface')
        result2 = facepp.detection.detect(img=File(pic2Path), mode = 'oneface')
        face_id1 = None
        face_id2 = None
        if len(result1['face']) > 0:
            face_id1 = result1['face'][0]['face_id']
        if len(result2['face']) > 0:
            face_id2 = result2['face'][0]['face_id']
        if face_id1 is not None and face_id2 is not None:
            #训练集label为1时就往group里创建两个名字一样的人
            if data['label'] == '1':
                facepp.person.create(group_name = groupName,face_id = [face_id1,face_id2])


'''
描述：初始化face++,默认创建test组
参数：[in]API_KEY,str
      [in]API_SECRET,str
      [out]api,face++接口实例
      [out]flag,,是否需要训练
'''
def initFacepp(API_KEY,API_SECRET):
    if API_KEY is None:
        API_KEY='4217654378594ee30144b2309472cc3a'
    if API_SECRET is None:
        API_SECRET='m7ZqIyi6rMcANlPhsgVnKfT7suV6o3lU'
    api = API(API_KEY, API_SECRET)

    flag = True
    #创建组
    flag = isNeedTrain(facepp = api)
    return api,flag

'''
描述：解析并读取数据集
参数：[in]dataListPath,样本集文件路径,str
      [out]dataSet,数据集,list,[{'pic1':'xxx','pic2':'xxx','label':'xxx'},....],dataSet['pic1'],dataSet['pic2']为两个图片的路径，dataSet['label']为标签
'''
def phraseDataList(dataListPath):
    dataFile = open(dataListPath.decode('utf-8'))
    dataSet = []
    try:
        list_of_all_the_lines = dataFile.readlines()
        for line in list_of_all_the_lines:
            #如果一行为换行符说明已到头，不继续解析
            if line == '\n':
                break
            #以:将字符串分割
            tmpList = line.split(':')
            tmpList[0] = tmpList[0].replace('/','\\')
            #得到第一幅图的绝对路径
            tmpList[0] = dataBasePath+tmpList[0]
            tmpList[1] = tmpList[1].replace('/','\\')
            #得到第二幅图的绝对路径
            tmpList[1] = dataBasePath+tmpList[1]
            tmpDict = {}
            tmpDict['pic1'] = tmpList[0]
            tmpDict['pic2'] = tmpList[1]
            tmpDict['label'] = tmpList[2].strip('\n')
            dataSet.append(tmpDict)
    finally:
        dataFile.close()
    return  dataSet


if __name__=='__main__':
    facepp,needTrain = initFacepp(None,None)
    if needTrain == True:
        train(dataBasePath+'datalist.txt')

    result = facepp.recognition.train(group_name = groupName, type = 'all')
    session_id = result['session_id']
    facepp.wait_async(session_id)

    #此处可以撸测试集,判断识别结果的人名是否一致
    result = facepp.recognition.recognize(img = File(dataBasePath+'pictures\\0088298_057.jpg'), group_name = groupName)
    pName1 = result['face'][0]['candidate'][0]['person_name']
    result = facepp.recognition.recognize(img = File(dataBasePath+'pictures\\0088298_019.jpg'), group_name = groupName)
    pName2 = result['face'][0]['candidate'][0]['person_name']
    if pName1 == pName2:
        print 1
    else:
        print 0
