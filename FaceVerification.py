# -*- coding: utf-8 -*-


dataBasePath = 'E:\\中兴捧月-人工智能-测试数据\\人工智能大赛-人脸认证\\人脸认证-第一期测试数据\\'

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
            #已:将字符串分割
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
        dataFile.close( )
    return  dataSet


if __name__=='__main__':
    dataSet = phraseDataList(dataBasePath+'datalist.txt')

    #测试phraseDataList是否正确
    if dataSet is not None:
        for dataLine in dataSet:
            print '--'*30
            print dataLine['pic1']
            print dataLine['pic2']
            print dataLine['label']
            print '--'*30