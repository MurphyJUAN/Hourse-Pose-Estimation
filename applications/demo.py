#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import vg
import copy

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/humpback_side.JPG'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

############HEAD SIDE########
IMAGE_HEAD_SIDE_NORMAL = PROJECT_PATH + '/data/images/stand_normal_side.JPG'
IMAGE_HEAD_SIDE_WRONG = PROJECT_PATH + '/data/images/humpback_side.JPG'
###########

##########HEAD FRONT##########
IMAGE_HEAD_FRONT_NORMAL = PROJECT_PATH + '/data/images/stand_normal.JPG'
IMAGE_HEAD_FRONT_WRONG = PROJECT_PATH + '/data/images/head_tilt.JPG'
#############

##########SPIN SIDE##########
IMAGE_SPIN_SIDE_NORMAL = PROJECT_PATH + '/data/images/stand_normal_side.JPG'
IMAGE_SPIN_SIDE_WRONG = PROJECT_PATH + '/data/images/humpback_murphy.JPG'
#############

##########KNEE SIDE##########
IMAGE_KNEE_SIDE_NORMAL = PROJECT_PATH + '/data/images/stand_normal_side.JPG'
IMAGE_KNEE_SIDE_WRONG = PROJECT_PATH + '/data/images/humpback_side_more2.JPG'
#############

##########HEAD TEST##########
IMAGE_HEAD_FRONT = PROJECT_PATH + '/data/images/左前傾(0).JPG'
IMAGE_HEAD_SIDE = PROJECT_PATH + '/data/images/左前傾(90).JPG'
#############

##########HEAD TEST##########
IMAGE_STAND_0 = PROJECT_PATH + '/data/images/立正(0)_3.JPG'
IMAGE_STAND_30 = PROJECT_PATH + '/data/images/立正(30)_3.JPG'
IMAGE_STAND_60 = PROJECT_PATH + '/data/images/立正(60)_3.JPG'
IMAGE_STAND_90 = PROJECT_PATH + '/data/images/立正(90)_3.JPG'
IMAGE_STAND_120 = PROJECT_PATH + '/data/images/立正(120)_3.JPG'
IMAGE_STAND_150 = PROJECT_PATH + '/data/images/立正(150)_3.JPG'
IMAGE_STAND_180 = PROJECT_PATH + '/data/images/立正(180)_3.JPG'
#############

##########HEAD TEST##########
IMAGE_HUMP_0 = PROJECT_PATH + '/data/images/駝背(0).JPG'
IMAGE_HUMP_30 = PROJECT_PATH + '/data/images/駝背(30).JPG'
IMAGE_HUMP_60 = PROJECT_PATH + '/data/images/駝背(60).JPG'
IMAGE_HUMP_90 = PROJECT_PATH + '/data/images/駝背(90).JPG'
IMAGE_HUMP_120 = PROJECT_PATH + '/data/images/駝背(120).JPG'
IMAGE_HUMP_150 = PROJECT_PATH + '/data/images/駝背(150).JPG'
IMAGE_HUMP_180 = PROJECT_PATH + '/data/images/駝背(180).JPG'
#############

###########HOURSE NORMAL######
HOUSE_STAND_1 = PROJECT_PATH + '/data/images/馬術正常1.JPG'
HOUSE_STAND_2 = PROJECT_PATH + '/data/images/馬術正常2.JPG'
HOUSE_STAND_3 = PROJECT_PATH + '/data/images/馬術正常3.JPG'
HOUSE_STAND_4 = PROJECT_PATH + '/data/images/馬術正常4.JPG'
HOUSE_STAND_5 = PROJECT_PATH + '/data/images/馬術正常5.JPG'
########

###########HOURSE HUMP######
HOUSE_HUMP_1 = PROJECT_PATH + '/data/images/馬術錯誤_2(1).JPG'
HOUSE_HUMP_2 = PROJECT_PATH + '/data/images/馬術錯誤_2(2).JPG'
HOUSE_HUMP_3 = PROJECT_PATH + '/data/images/馬術錯誤_2(3).JPG'
HOUSE_HUMP_4 = PROJECT_PATH + '/data/images/馬術錯誤_2(4).JPG'
HOUSE_HUMP_5 = PROJECT_PATH + '/data/images/馬術錯誤_2(5).JPG'
HOUSE_HUMP_6 = PROJECT_PATH + '/data/images/馬術錯誤_2(6).JPG'
########

TEST_IMAGE_LIST = [
    # IMAGE_HEAD_FRONT_NORMAL,
    # IMAGE_HEAD_FRONT_WRONG,
    # IMAGE_HEAD_SIDE_NORMAL,
    # IMAGE_HEAD_SIDE_WRONG,
    # IMAGE_SPIN_SIDE_NORMAL,
    # IMAGE_SPIN_SIDE_WRONG,
    # IMAGE_KNEE_SIDE_NORMAL,
    # IMAGE_KNEE_SIDE_WRONG,
    # IMAGE_HEAD_FRONT,
    # IMAGE_HEAD_SIDE,
    # IMAGE_STAND_0,
    # IMAGE_STAND_30,
    # IMAGE_STAND_60,
    # IMAGE_STAND_90,
    # IMAGE_STAND_120,
    # IMAGE_STAND_150,
    # IMAGE_STAND_180,
    # IMAGE_HUMP_0,
    # IMAGE_HUMP_30,
    # IMAGE_HUMP_60,
    # IMAGE_HUMP_90,
    # IMAGE_HUMP_120,
    # IMAGE_HUMP_150,
    # IMAGE_HUMP_180,
    # HOUSE_STAND_1,
    # HOUSE_STAND_2,
    # HOUSE_STAND_3,
    # HOUSE_STAND_4,
    # HOUSE_STAND_5,
    # HOUSE_HUMP_1,
    # HOUSE_HUMP_2,
    # HOUSE_HUMP_3,
    HOUSE_HUMP_4,
    HOUSE_HUMP_5,
    HOUSE_HUMP_6
]

def main():
    for i in TEST_IMAGE_LIST:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

        # create pose estimator
        image_size = image.shape
        pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

        # load model
        pose_estimator.initialise()

        try:
            # estimation
            print('START')
            pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
            print('=================================')
            print(str(i))
            print('pose 3d:', pose_3d)
            print('pose_2d:', pose_2d)
            angle_target_2d = pose_2d[0].copy()
            angle_target_3d = pose_3d[0].copy()

            
            #投影頭部的點到 x-z 平面
            angle_2d = calculate_angle_2d([angle_target_3d[0][8],angle_target_3d[2][8]],
                                [angle_target_3d[0][10], angle_target_3d[2][10]])
            print(angle_2d)

            #3D頭部
            standard_v = np.array([0,0,-1])
            v_t = np.array([angle_target_3d[0][8] - angle_target_3d[0][10], 
                angle_target_3d[1][8] - angle_target_3d[1][10],
                angle_target_3d[2][8] - angle_target_3d[2][10]])
            print('====頭的角度===', vg.angle(standard_v, v_t))
            #3D肩膀
            vector = np.array([abs(angle_target_3d[0][6] - angle_target_3d[0][3]), \
            abs(angle_target_3d[1][6] - angle_target_3d[1][3]),\
            abs(angle_target_3d[2][6] - angle_target_3d[2][3])])\

            if vector[0]>vector[1] and vector[0]>vector[2]:
                standard_v = np.array([1, 0 ,0])
            elif vector[1]>vector[0] and vector[1]>vector[2]:
                standard_v = np.array([0, -1 ,0])
            else:
                print('判斷肩膀正面側面出現問題！預設正面')
                standard_v = np.array([1, 0 ,0])
            v_t = np.array([angle_target_3d[0][11] - angle_target_3d[0][14], 
                angle_target_3d[1][11] - angle_target_3d[1][14],
                angle_target_3d[2][11] - angle_target_3d[2][14]])
            print('====肩膀的角度===', vg.angle(standard_v, v_t))
            #3D脊椎(脊椎中間連線頸椎中間與脊椎中間連線屁股中間)
            # standard_v = np.array([angle_target_3d[0][7] - angle_target_3d[0][8], 
            #     angle_target_3d[1][7] - angle_target_3d[1][8],
            #     angle_target_3d[2][7] - angle_target_3d[2][8]])
            # v_t = np.array([angle_target_3d[0][7] - angle_target_3d[0][0], 
            #     angle_target_3d[1][7] - angle_target_3d[1][0],
            #     angle_target_3d[2][7] - angle_target_3d[2][0]])
            # print('====脊椎的角度===', vg.angle(standard_v, v_t))

            #3D脊椎(雙腳中間連線脊椎終點與脊椎中間連線頸椎中點)
            # foot_center_x = (angle_target_3d[0][3] + angle_target_3d[0][6])/2
            # foot_center_y = (angle_target_3d[1][3] + angle_target_3d[1][6])/2
            # foot_center_z = (angle_target_3d[2][3] + angle_target_3d[2][6])/2
            # standard_v = np.array([foot_center_x - angle_target_3d[0][0], 
            #     foot_center_y - angle_target_3d[1][0],
            #     foot_center_z - angle_target_3d[2][0]])
            # v_t = np.array([angle_target_3d[0][8] - angle_target_3d[0][0], 
            #     angle_target_3d[1][8] - angle_target_3d[1][0],
            #     angle_target_3d[2][8] - angle_target_3d[2][0]])
            # print('====脊椎的角度===', vg.angle(standard_v, v_t))

            #馬術的3D脊椎(雙腳中間連線脊椎終點與脊椎中間連線頸椎中點)
            standard_v = np.array([0,0,1])
            v_t = np.array([angle_target_3d[0][8] - angle_target_3d[0][0], 
                angle_target_3d[1][8] - angle_target_3d[1][0],
                angle_target_3d[2][8] - angle_target_3d[2][0]])
            print('====脊椎的角度===', vg.angle(standard_v, v_t))

            #3D左膝蓋
            standard_v = np.array([angle_target_3d[0][5] - angle_target_3d[0][4], 
                angle_target_3d[1][5] - angle_target_3d[1][4],
                angle_target_3d[2][5] - angle_target_3d[2][4]])
            v_t = np.array([angle_target_3d[0][5] - angle_target_3d[0][6], 
                angle_target_3d[1][5] - angle_target_3d[1][6],
                angle_target_3d[2][5] - angle_target_3d[2][6]])
            print('====左膝蓋的角度===', vg.angle(standard_v, v_t))
            #3D右膝蓋
            standard_v = np.array([angle_target_3d[0][2] - angle_target_3d[0][1], 
                angle_target_3d[1][2] - angle_target_3d[1][1],
                angle_target_3d[2][2] - angle_target_3d[2][1]])
            v_t = np.array([angle_target_3d[0][2] - angle_target_3d[0][3], 
                angle_target_3d[1][2] - angle_target_3d[1][3],
                angle_target_3d[2][2] - angle_target_3d[2][3]])
            print('====右膝蓋的角度===', vg.angle(standard_v, v_t))

            #投影法

            # Show 2D and 3D poses

            display_results(image, pose_2d, visibility, pose_3d)
        except ValueError:
            print('No visible people in the image. Change CENTER_TR in packages/lifting/utils/config.py ...')

        # close model
        pose_estimator.close()



def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def calculate_angle_2d(p1, p2):
    angle = math.atan2(- int(p2[0]) + int(p1[0]), int(p2[1]) - int(p1[1])) * 180.0 / np.pi
    return angle

def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
