# -*- coding: utf-8 -*-
from facepp import API
from facepp import File
from numpy import *
import string
import operator
import os
import time


dataBasePath = 'E:\\code\\Python\\FaceVerification\\data\\'
groupName = 'alwTestalw'
trainSetPath = dataBasePath+'trainSet.txt'
trainLabelPath = dataBasePath+'trainLabel.txt'

result_dict = {'Male':0,'Female':500,'Asian':100,'White':400,'Black':700,'None':100,'Dark':400,'Normal':700}



'''
描述：是否需要训练
参数：[out]True,需要训练，False，不需要训练
'''
def isNeedTrain():
    if os.path.exists(trainSetPath) == True and os.path.exists(trainLabelPath) == True:
        return False
    return True



def createTrainVec(detectResult1,detectResult2,landmark1,landmark2):
    vec = []
    if len(detectResult1['face']) > 0 and len(detectResult2['face']) > 0:
        vec.append(detectResult1['face'][0]['attribute']['age']['range'])
        vec.append(detectResult1['face'][0]['attribute']['age']['value'])
        vec.append(result_dict[detectResult1['face'][0]['attribute']['gender']['value']])
        vec.append(result_dict[detectResult1['face'][0]['attribute']['glass']['value']])
        vec.append(detectResult1['face'][0]['attribute']['pose']['pitch_angle']['value'])
        vec.append(detectResult1['face'][0]['attribute']['pose']['roll_angle']['value'])
        vec.append(detectResult1['face'][0]['attribute']['pose']['yaw_angle']['value'])
        vec.append(result_dict[detectResult1['face'][0]['attribute']['race']['value']])
        vec.append(detectResult1['face'][0]['attribute']['smiling']['value'])
        #记录相对于contour_chin的相对坐标
        base1_x = landmark1['result'][0]['landmark']['contour_chin']['x']
        base1_y = landmark1['result'][0]['landmark']['contour_chin']['y']
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left4']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left4']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left5']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left5']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left6']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left6']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left7']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left7']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left8']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left8']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left9']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['contour_left9']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_bottom']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_bottom']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_center']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_center']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_left_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_left_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_lower_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_lower_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_lower_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_lower_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_pupil']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_pupil']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_right_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_right_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_top']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_top']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_upper_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_upper_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_upper_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eye_upper_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_left_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_left_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_lower_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_lower_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_lower_middle']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_lower_middle']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_lower_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_lower_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_right_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_right_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_upper_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_upper_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_upper_middle']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_upper_middle']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_upper_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['left_eyebrow_upper_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_left_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_left_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_bottom']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_bottom']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_left_contour1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_left_contour1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_left_contour2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_left_contour2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_left_contour3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_left_contour3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_right_contour1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_right_contour1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_right_contour2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_right_contour2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_right_contour3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_right_contour3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_top']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_lower_lip_top']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_right_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_right_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_bottom']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_bottom']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_left_contour1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_left_contour1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_left_contour2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_left_contour2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_left_contour3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_left_contour3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_right_contour1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_right_contour1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_right_contour2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_right_contour2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_right_contour3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_right_contour3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_top']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['mouth_upper_lip_top']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_left1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_left1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_left2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_left2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_left3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_left3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_lower_middle']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_lower_middle']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_right1']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_right1']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_right2']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_right2']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_right3']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_contour_right3']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_left']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_left']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_right']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_right']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_tip']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['nose_tip']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_bottom']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_bottom']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_center']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_center']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_left_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_left_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_lower_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_lower_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_lower_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_lower_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_pupil']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_pupil']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_right_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_right_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_top']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_top']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_upper_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_upper_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_upper_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eye_upper_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_left_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_left_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_lower_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_lower_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_lower_middle']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_lower_middle']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_lower_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_lower_right_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_right_corner']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_right_corner']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_upper_left_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_upper_left_quarter']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_upper_middle']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_upper_middle']['y'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_upper_right_quarter']['x'])
        vec.append(base1_x-landmark1['result'][0]['landmark']['right_eyebrow_upper_right_quarter']['y'])


        vec.append(detectResult2['face'][0]['attribute']['age']['range'])
        vec.append(detectResult2['face'][0]['attribute']['age']['value'])
        vec.append(result_dict[detectResult2['face'][0]['attribute']['gender']['value']])
        vec.append(result_dict[detectResult2['face'][0]['attribute']['glass']['value']])
        vec.append(detectResult2['face'][0]['attribute']['pose']['pitch_angle']['value'])
        vec.append(detectResult2['face'][0]['attribute']['pose']['roll_angle']['value'])
        vec.append(detectResult2['face'][0]['attribute']['pose']['yaw_angle']['value'])
        vec.append(result_dict[detectResult2['face'][0]['attribute']['race']['value']])
        vec.append(detectResult2['face'][0]['attribute']['smiling']['value'])
        #记录相对于contour_chin的相对坐标
        base2_x = landmark2['result'][0]['landmark']['contour_chin']['x']
        base2_y = landmark2['result'][0]['landmark']['contour_chin']['y']
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left4']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left4']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left5']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left5']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left6']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left6']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left7']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left7']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left8']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left8']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left9']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['contour_left9']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_bottom']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_bottom']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_center']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_center']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_left_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_left_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_lower_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_lower_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_lower_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_lower_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_pupil']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_pupil']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_right_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_right_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_top']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_top']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_upper_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_upper_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_upper_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eye_upper_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_left_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_left_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_lower_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_lower_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_lower_middle']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_lower_middle']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_lower_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_lower_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_right_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_right_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_upper_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_upper_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_upper_middle']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_upper_middle']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_upper_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['left_eyebrow_upper_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_left_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_left_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_bottom']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_bottom']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_left_contour1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_left_contour1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_left_contour2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_left_contour2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_left_contour3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_left_contour3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_right_contour1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_right_contour1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_right_contour2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_right_contour2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_right_contour3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_right_contour3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_top']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_lower_lip_top']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_right_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_right_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_bottom']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_bottom']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_left_contour1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_left_contour1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_left_contour2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_left_contour2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_left_contour3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_left_contour3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_right_contour1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_right_contour1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_right_contour2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_right_contour2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_right_contour3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_right_contour3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_top']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['mouth_upper_lip_top']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_left1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_left1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_left2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_left2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_left3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_left3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_lower_middle']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_lower_middle']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_right1']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_right1']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_right2']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_right2']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_right3']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_contour_right3']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_left']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_left']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_right']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_right']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_tip']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['nose_tip']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_bottom']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_bottom']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_center']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_center']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_left_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_left_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_lower_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_lower_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_lower_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_lower_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_pupil']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_pupil']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_right_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_right_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_top']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_top']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_upper_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_upper_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_upper_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eye_upper_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_left_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_left_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_lower_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_lower_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_lower_middle']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_lower_middle']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_lower_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_lower_right_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_right_corner']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_right_corner']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_upper_left_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_upper_left_quarter']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_upper_middle']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_upper_middle']['y'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_upper_right_quarter']['x'])
        vec.append(base2_x-landmark2['result'][0]['landmark']['right_eyebrow_upper_right_quarter']['y'])

    if len(vec) > 0:
        return vec
    return None

'''
将数据集中的特征值归一化
'''
def autoNormalize(dataSet):
    minVal = dataSet.min(0)                     #最小的列值
    maxVal = dataSet.max(0)                     #最大的列值
    ranges = maxVal - minVal
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVal,(m,1))
    normalDataSet = normalDataSet/tile(ranges,(m,1))
    return normalDataSet

'''
描述：dump模型文件
'''
def dumpTrainFile(trainSet):
    fl=open(trainSetPath, 'w')
    for line in trainSet:
        for value in line:
            fl.write(str(value))
            fl.write(',')
        fl.write('\n')
    fl.close()

'''
描述：dump标签文件
'''
def dumpLabelFile(trainLabel):
    fl=open(trainLabelPath, 'w')
    for label in trainLabel:
        fl.write(str(label))
        fl.write(',')
    fl.close()

'''
描述：读取模型文件
'''
def loadTrainFile():
    trainSet = []
    fl = open(trainSetPath, 'r')
    lines = fl.readlines()
    for line in lines:
        line.strip('\n')
        line.strip(',')
        tl = line.split(',')
        if tl[len(tl)-1] == '\n':
            tl = tl[:len(tl)-1]
        tl = [float(e) for e in tl]
        trainSet.append(tl)
    fl.close()
    return array(trainSet)

'''
描述：读取标签文件
'''
def loadLabelFile():
    trainLabel = []
    fl = open(trainLabelPath, 'r')
    lines = fl.readline()
    lines.strip('\n')
    lines.strip(',')
    tl = lines.split(',')
    if tl[len(tl)-1] == '':
        tl = tl[:len(tl)-1]
    tl = [int(e) for e in tl]
    trainLabel = tl
    fl.close()
    return trainLabel

'''
描述：训练数据集
参数：[in]trainDataPath,str,训练数据集的路径
'''
def train(facepp,trainDataPath):
    trainSet = []
    trainLabel = []
    tmpCount = 0
    #获得人名与图片路径字典

    personDict = {}
    for parent,dirnames,filenames in os.walk(trainDataPath):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for dirname in dirnames:
            pathList = []
            for parent2,dirnames2,filenames2 in os.walk(trainDataPath+'\\'+dirname):
                for filename2 in filenames2:
                    pathList.append(trainDataPath+'\\'+dirname+'\\'+filename2)
            personDict[dirname] = pathList


    #生成正样本和标签集
    for personList in personDict.values():
        if len(personList) > 1:
            i = 0
            while i <= len(personList)-1:
                j = i+1
                while j <= len(personList)-1:
                    result1 = facepp.detection.detect(img = File(personList[i]),attribute = ['gender', 'age', 'race', 'smiling', 'glass', 'pose'])
                    result2 = facepp.detection.detect(img = File(personList[j]),attribute = ['gender', 'age', 'race', 'smiling', 'glass', 'pose'])
                    if len(result1['face']) > 0 and len(result2['face']) > 0:
                        faceid1 = result1['face'][0]['face_id']
                        faceid2 = result2['face'][0]['face_id']
                        landmark1 = facepp.detection.landmark(face_id = faceid1)
                        landmrak2 = facepp.detection.landmark(face_id = faceid2)
                        vec = createTrainVec(result1,result2,landmark1,landmrak2)
                        if vec is not None:
                            trainSet.append(vec)
                            trainLabel.append('1')
                    j+=1
                    tmpCount = tmpCount+1
                    print tmpCount
                    if tmpCount%1000 == 0:
                        print 'sleep 10s in postive'
                        time.sleep(10)
                i+=1

    #生成负样本和标签集
    personList = personDict.values()
    personList = personList[:len(personList)/8]
    i = 0
    while i < len(personList):
        j = i+1
        while j < len(personList):
            k = 0
            while k < len(personList[i]):
                p = 0
                while p < len(personList[j]):
                    result1 = facepp.detection.detect(img = File(personList[i][k]),attribute = ['gender', 'age', 'race', 'smiling', 'glass', 'pose'])
                    result2 = facepp.detection.detect(img = File(personList[j][p]),attribute = ['gender', 'age', 'race', 'smiling', 'glass', 'pose'])
                    if len(result1['face']) > 0 and len(result2['face']) > 0:
                        faceid1 = result1['face'][0]['face_id']
                        faceid2 = result2['face'][0]['face_id']
                        landmark1 = facepp.detection.landmark(face_id = faceid1)
                        landmrak2 = facepp.detection.landmark(face_id = faceid2)
                        vec = createTrainVec(result1,result2,landmark1,landmark2)
                        vec = createTrainVec(result1,result2)
                        if vec is not None:
                            trainSet.append(vec)
                            trainLabel.append('0')
                    p+=1
                    tmpCount+=1
                    print tmpCount
                    if tmpCount%1000 == 0:
                        print 'sleep 10s in negtive'
                        time.sleep(10)
                k+=1
            j+=1
        i+=1
    dumpTrainFile(trainSet)
    dumpLabelFile(trainLabel)

    return array(trainSet),array(trainLabel)

'''
kNN算法分类器
inX:测试向量
dataSet:数据向量（不包含标签）
labels:对应
'''
def kNNClassifyer(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]              #数据集行数
    #tile(inX,(dataSetSize,1))表示将一个inX行向量拓展成dataSetSize行的矩阵，并用inX填充
    #diffMat为inX与各行dataSet之间特征的差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2                      #将diffMat中的元素算个平方
    #sqDistances是行向量，sqDiffMat.sum(axis=1)表示把第i行sqDiffMat的值做加法放在sqDistances[i]的位置
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5                #开方得到距离
    sortedDistIndicies = distances.argsort()    #升序排序并将索引返回给sortedDistIndicies
    classCount = {}
    #计算与inX最近的K个dataSet的标签并保存至classCount
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1

    #对classCount降序排序，票数最多的标签为预测结果
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    return int(sortedClassCount[0][0])



'''
描述：初始化face++
参数：[in]API_KEY,str
      [in]API_SECRET,str
      [out]api,face++接口实例
'''
def initFacepp(API_KEY,API_SECRET):
    if API_KEY is None:
        API_KEY='4217654378594ee30144b2309472cc3a'
    if API_SECRET is None:
        API_SECRET='m7ZqIyi6rMcANlPhsgVnKfT7suV6o3lU'
    api = API(API_KEY, API_SECRET)
    return api

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
    facepp = initFacepp(None,None)
    trainSet = None
    trainLabel = None
    if isNeedTrain():
        trainSet,trainLabel = train(trainDataPath=dataBasePath+'trainPicture2',facepp = facepp)
    else:
        trainSet = loadTrainFile()
        trainLabel = loadLabelFile()
    dataSet = phraseDataList(dataBasePath+'datalist.txt')
    for data in dataSet:
        testResult = 0
        result1 = facepp.detection.detect(img = File(data['pic1']),attribute = ['gender', 'age', 'race', 'smiling', 'glass', 'pose'])
        result2 = facepp.detection.detect(img = File(data['pic2']),attribute = ['gender', 'age', 'race', 'smiling', 'glass', 'pose'])
        vec = createTrainVec(result1,result2)
        start = time.clock()
        if vec is not None:
            testResult = kNNClassifyer(vec, trainSet, trainLabel, 5)
        end = time.clock()
        if testResult == int(data['label']):
            count+=1
        print 'test result:%d fact result:%s time:%.03f' % (testResult,data['label'], end-start)
    print 'correct count:'+str(count)+' correct ratio:'+str(count/1000.0)