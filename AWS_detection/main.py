import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten
from tensorflow.keras.layers import Dense, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

from alibi_detect.saving import load_detector

import os
from glob import glob

# from lib.VAEmodel import _Load_model
from lib.read_txt import _Read_coordinate
from lib.ROI import _XY2ROI
from lib.perspective_trans import _Ptrans
import lib.param as P

from yolov5.detect import run
'''
# Detect cars
run(weights=P._source_car_weights_dir, source=P._source_img_dir, save_txt=P._save_txt,
    classes=P._classes, nosave=P._nosave, name=P._save_car_name)

# Detect road marks
run(weights=P._source_roadmark_weights_dir, source=P._source_img_dir, save_txt=P._save_txt,
    nosave=P._nosave, name=P._save_road_mark_name, conf_thres=P._conf_thres)
'''
# Get bboxes coordinates
car_bbox_dir = os.getcwd()+P._detect_result_dir+P._save_car_name+'\\labels\\'
car_bbox = os.listdir(car_bbox_dir)
bbox_car_xy = _Read_coordinate(file_path=car_bbox_dir+car_bbox[0], img_h=P._img_h, img_w=P._img_w)

roadmark_bbox_dir = os.getcwd()+P._detect_result_dir+P._save_road_mark_name+'\\labels\\'
roadmark_bbox = os.listdir(roadmark_bbox_dir)
bbox_roadmark_xy = _Read_coordinate(file_path=roadmark_bbox_dir+roadmark_bbox[0], img_h=P._img_h, img_w=P._img_w, road_mark=True)

# Determine area for detecting
road_area = _XY2ROI(bbox_car_xy, bbox_roadmark_xy)

'''
# check road area is appropriate
img = cv2.imread(_source_img_dir, cv2.IMREAD_COLOR)
mask = np.zeros((P._img_h, P._img_w, 3), np.int32)
road_area = np.array(road_area, np.int32)
cv2.fillPoly(mask, road_area, (255, 255, 255))
masked_img = cv2.bitwise_and(np.array(img, np.uint8), np.array(mask, np.uint8))
cv2.imshow('mask', masked_img)
cv2.waitKey(0)
'''

# Perspective transforming to road areas for VAE
road_area_Ptrans = _Ptrans(road_area)

# VAE for detecting obstacles
# filepath = 'C:/Users/pc/Desktop/road/new/'
od = load_detector(P._VAE_weights_dir)

test_pred = od.predict(
    road_area_Ptrans,
    outlier_type='instance',
    return_feature_score=True,
    return_instance_score=True
)

# target = np.zeros(test_img.shape[0],).astype(int)
# labels = ['normal', 'outlier']
# plot_instance_score(test_pred, target, labels, 0.002)

# print(test_pred['data']['is_outlier'])


# result is the road lane that obstacles are on
#print(result)
result = test_pred['data']['is_outlier']
for r in range(len(result)):
    if result[r]:
        save_img = cv2.cvtColor(road_area_Ptrans[r], cv2.COLOR_BGR2RGB) * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite(P._detect_img_save + 'obstacle' + str(r) + '.png', cv2.resize(save_img, (512, 512)))
    else:
        print('no obstacle')
'''
for r in result:
    save_img = cv2.cvtColor(road_img[r], cv2.COLOR_BGR2RGB) * 255
    save_img = save_img.astype(np.uint8)
    cv2.imwrite(P._detect_img_save+'obstacle'+str(r)+'.png', cv2.resize(save_img, (512, 512)))
'''
'''
for r in range(len(road_img)):
    if r in result:
        cv2.imshow('road_img'+str(r), cv2.resize(road_img[r], (512, 512)))
        cv2.imshow('pred'+str(r), cv2.resize(pred[r], (512, 512)))
        print(road_img[r].shape)

        #cv2.imshow('sub'+str(r), cv2.resize(pred[i]-road_img[r], (512, 512)))
cv2.waitKey(0)
'''