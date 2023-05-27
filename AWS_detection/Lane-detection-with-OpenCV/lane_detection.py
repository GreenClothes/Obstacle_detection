import numpy as np, cv2
from function.preprocessing import *
from function.detect import *

CANNY_THRESHOLD1 = 130
CANNY_THRESHOLD2 = 200

image = cv2.imread('images/road_detect.jpg')
if image is None: raise Exception('영상파일 읽기 오류')

Canny_img = cv2.Canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

#cv2.imshow('image', image)
#cv2.imshow('Canny_img', Canny_img)
#cv2.waitKey(0)

ROI_img = ROI(Canny_img)

#cv2.imshow('image', image)
#cv2.imshow('ROI_img', ROI_img)
#cv2.waitKey(0)

PERSPECTIVE_img, inverse_mtrx = PERSPECTIVE_TRANS(ROI_img)

#cv2.imshow('image', image)
#cv2.imshow('Perspective transform image', PERSPECTIVE_img)
#cv2.waitKey(0)

lfit, rfit = sliding_window(PERSPECTIVE_img)
#cv2.imshow('perspective img', PERSPECTIVE_img)
#cv2.waitKey(0)

lane_detect, l_pts, r_pts = draw_window(image, PERSPECTIVE_img, inverse_mtrx, lfit, rfit)

#cv2.imshow('image', image)
#cv2.imshow('perspect', PERSPECTIVE_img)
#cv2.imshow('wind_perspect', lane_detect)
#cv2.waitKey(0)