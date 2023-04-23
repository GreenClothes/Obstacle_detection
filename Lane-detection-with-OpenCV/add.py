import numpy as np, cv2
from function.preprocessing import *
from function.detect import *
from function.add_func import *

CANNY_THRESHOLD1 = 130
CANNY_THRESHOLD2 = 200

image = cv2.imread('images/road1.jpg')
if image is None: raise Exception('image 파일 읽기 오류')

Canny_img = cv2.Canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

ROI_img = ROI(Canny_img)

PERSPECTIVE_img, inverse_mtrx = PERSPECTIVE_TRANS(ROI_img)

lfit, rfit = sliding_window(PERSPECTIVE_img)

lane_detect, l_pts, r_pts = draw_window(image, PERSPECTIVE_img, inverse_mtrx, lfit, rfit)

offset_img, mid_point = Center_line_offset(lane_detect, l_pts, r_pts, inverse_mtrx)

handle = cv2.imread('images/handle.png', cv2.IMREAD_COLOR)
if handle is None: raise Exception('handle 파일 읽기 오류')

handle = cv2.resize(handle, (300, 300))

theta = steering_wheel(offset_img, handle, mid_point)
cv2.imshow('steering wheel image', theta)
cv2.waitKey(0)