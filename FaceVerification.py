# -*- coding: utf-8 -*-
from facepp import API
from facepp import File
import os
import time

dataBasePath = 'E:\\code\\Python\\FaceVerification\\data\\'
groupName = 'alwTestalw'


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
    personCount = 0
    for parent,dirnames,filenames in os.walk(trainDataPath):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for dirname in dirnames:
            personCount+=1
            if personCount > 100:
                break
            faceList = []
            for parent2,dirnames2,filenames2 in os.walk(trainDataPath+'\\'+dirname):
                for filename2 in filenames2:
                    result = facepp.detection.detect(img=File(trainDataPath+'\\'+dirname+'\\'+filename2), mode = 'oneface')
                    if len(result['face']) > 0:
                        faceList.append(result['face'][0]['face_id'])
            if len(faceList)>0:
                facepp.person.create(face_id = faceList,group_name = groupName,person_name = dirname)
    session_id = facepp.train.identify(group_name = groupName)
    facepp.wait_async(session_id['session_id'])



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

count = 0   #正确次数
if __name__=='__main__':
    facepp,needTrain = initFacepp(None,None)  
    if needTrain == True:
        train(dataBasePath+'trainPicutre')
    dataSet = phraseDataList(dataBasePath+'datalist.txt')
    for data in dataSet:
        start = time.clock()
        result1 = facepp.recognition.identify(img = File(data['pic1']),group_name = groupName)
        result2 = facepp.recognition.identify(img = File(data['pic2']),group_name = groupName)
        flag = False
        realResult = '0'
        confidence = 0
        if len(result1['face'])>0 and len(result2['face'])>0:
            for personN1 in result1['face'][0]['candidate']:
                for personN2 in result2['face'][0]['candidate']:
                    if personN1['person_name'] == personN2['person_name']:
                        confidence = personN1['confidence']+50.0
                        flag = True
                        break
        end = time.clock()
        if flag == True:
            realResult = '1'
        if realResult == data['label']:
            count+=1
        print 'tset result:%s fact result:%s confidence:%f time:%.03f' % (realResult,data['label'], confidence, end-start)
        #print 'test result:'+realResult+' fact result:'+data['label']+' confidence:'+str(confidence)+' time:'+str(end-start)
    print 'correct count:'+str(count)+' correct ratio:'+str(count/6000.0)