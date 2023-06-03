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

import os
from glob import glob

from lib.VAEmodel import _Load_model
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
    nosave=P._nosave, name=P._save_road_mark_name)
'''
# Get bboxes coordinates
car_bbox_dir = os.getcwd()+P._detect_result_dir+P._save_car_name+'\\labels\\'
car_bbox = os.listdir(car_bbox_dir)
bbox_car_xy = _Read_coordinate(file_path=car_bbox_dir+car_bbox[0], img_h=P._img_h, img_w=P._img_w)

roadmark_bbox_dir = os.getcwd()+P._detect_result_dir+P._save_road_mark_name+'\\labels\\'
roadmark_bbox = os.listdir(roadmark_bbox_dir)
bbox_roadmark_xy = _Read_coordinate(file_path=roadmark_bbox_dir+roadmark_bbox[0], img_h=P._img_h, img_w=P._img_w)

# Determine area for detecting
road_area = _XY2ROI(bbox_car_xy, bbox_roadmark_xy)
'''
# Perspective transforming to road areas for VAE
road_area_Ptrans = _Ptrans(road_area)

# VAE for detecting obstacles
VAE = _Load_model()
VAE.load_weights(P._VAE_weights_dir)
road_img = np.array(road_area_Ptrans)
road_img = road_img.astype(np.float32) / 255.
pred = VAE.predict(road_img)
result = []

for i in range(road_img.shape[0]):
    r, g, b = cv2.split(road_area_Ptrans[i]-pred[i])
    img_flat = (r+g+b).flatten()
    if np.percentile(img_flat, 80, method='higher') >= P._VAE_threshold:
        result.append(i)

# result is the road lane that obstacles are on
print(result)
'''
'''
for r in range(len(road_img)):
    if not(r in result):
        cv2.imshow('road_img'+str(r), cv2.resize(road_img[r], (512, 512)))
        cv2.imshow('pred'+str(r), cv2.resize(pred[r], (512, 512)))
cv2.waitKey(0)
'''