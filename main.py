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
import lib.param as P

from yolov5.detect import run
'''
# Detect cars
run(weights=P._source_car_weights_dir, source=P._source_img_dir, save_txt=P._save_txt,
    classes=P._classes, nosave=P._nosave, name=P._save_car_name)

# Detect road marks
run(weights=P._source_roadmark_weights_dir, source=P._source_img_dir, save_txt=P._save_txt,
    classes=P._classes, nosave=P._nosave, name=P._save_road_mark_name)
'''
# Get bboxes coordinates
car_bbox_dir = os.getcwd()+P._detect_result_dir+P._save_car_name+'\\labels\\'
car_bbox = os.listdir(car_bbox_dir)
bbox_car_xy = _Read_coordinate(file_path=car_bbox_dir+car_bbox[0], img_h=P._img_h, img_w=P._img_w)

roadmark_bbox_dir = os.getcwd()+P._detect_result_dir+P._save_road_mark_name+'\\labels\\'
roadmark_bbox = os.listdir(roadmark_bbox_dir)
bbox_roadmark_xy = _Read_coordinate(file_path=roadmark_bbox_dir+roadmark_bbox[0], img_h=P._img_h, img_w=P._img_w)

# Determine area for detecting
