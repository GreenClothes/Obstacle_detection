import cv2, numpy as np
from param import _ROI, _img_w, _img_h, _lane_classes

def _XY2ROI(car, roadmark, img):
    ROI = [[int(_ROI[i][0]*_img_w), int(_ROI[i][1]*_img_h)] for i in range(len(_ROI))]

    roadmark_bbox = []
    lane_bbox = []

    # Dividing into lanes and roadmarks
    for i in range(len(roadmark)):
        if roadmark[i][0][0] in _lane_classes:
            lane_bbox.append(roadmark[i][1:])
        else:
            roadmark_bbox.append(roadmark[i][1:])

    # Get road lane area

def _get_road_lane(roi, lane):
    # 먼저 인식된 차선이 ROI 안에 있는지부터 확인
    road_middle = (max(roi[0]) - min(roi[0])) // 2

    for i in range(len(lane)):
        lane_middle = (lane[i][0][0] - lane[i][1][0]) // 2
        if road_middle > lane_middle:
            # remove lane from roi and divide area
            lane_remove = [[lane[i][1][0]-0.05*_img_w, lane[i][1][1]],
                           [lane[i][1][0]+0.05*_img_w, lane[i][1][1]],
                           [lane[i][3][0]-0.05*_img_w, lane[i][3][1]],
                           [lane[i][3][0]+0.05*_img_w, lane[i][3][1]]]
        else:
            # remove lane from roi and divide area
            lane_remove = [[lane[i][0][0]-0.05*_img_w, lane[i][0][1]],
                           [lane[i][0][0]+0.05*_img_w, lane[i][0][1]],
                           [lane[i][2][0]-0.05*_img_w, lane[i][2][1]],
                           [lane[i][2][0]+0.05*_img_w, lane[i][2][1]]]
