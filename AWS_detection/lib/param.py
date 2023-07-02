import os
import numpy as np

_ROOT = os.getcwd()
_source_img_dir = _ROOT+'\\yolov5\\data\\images\\detect\\test.png'
_source_car_weights_dir = _ROOT+'\\yolov5\\weights\\yolov5s.pt'
_source_roadmark_weights_dir = _ROOT+'\\yolov5\\weights\\best.pt'
_save_txt = True
_nosave = False
_classes = [2, 3, 5, 7] # coco128.yaml (2 : car / 3 : motorcycle / 5 : bus / 7 : truck)
_conf_thres = 0.4
_save_car_name = 'car1'
_save_road_mark_name = 'roadmark1'
_detect_result_dir = '\\yolov5\\runs\\detect\\'
_img_h = 1080
_img_w = 1920
_ROI = np.array([[0, _img_h*0.52], [_img_w, _img_h*0.52], [_img_w, _img_h*0.65], [0, _img_h*0.65]])
_lane_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # road lane classes
# _VAE_weights_dir = 'C:/Users/pc/Desktop/road/weights/checkpoint200/'
_VAE_weights_dir = 'C:/Users/pc/Desktop/road/new/'
# _VAE_threshold = 0.08
_detect_img_save = _ROOT+'\\Obstacle\\'