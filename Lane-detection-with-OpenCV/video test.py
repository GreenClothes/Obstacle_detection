import numpy as np, cv2
from function.preprocessing import *
from function.detect import *
from function.add_func import *

CANNY_THRESHOLD1 = 130
CANNY_THRESHOLD2 = 200

cap = cv2.VideoCapture('images/drive.mp4')

handle = cv2.imread('images/handle.png', cv2.IMREAD_COLOR)
if handle is None: raise Exception('handle 파일 읽기 오류')

handle = cv2.resize(handle, (300, 300))

# 영상 저장 설정
#cap_size = (int(cap.get(3)), int(cap.get(4)))
#fps = 25
#fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#out = cv2.VideoWriter('result.avi', fcc, fps, cap_size)

while (1):
    ret, image = cap.read()
    if ret:

        Canny_img = cv2.Canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

        ROI_img = ROI(Canny_img)

        PERSPECTIVE_img, inverse_mtrx = PERSPECTIVE_TRANS(ROI_img)

        lfit, rfit = sliding_window(PERSPECTIVE_img)

        lane_detect, l_pts, r_pts = draw_window(image, PERSPECTIVE_img, inverse_mtrx, lfit, rfit)

        offset_img, mid_point = Center_line_offset(lane_detect, l_pts, r_pts, inverse_mtrx)

        wheel = steering_wheel(offset_img, handle, mid_point)

        # 영상 저장
        #out.write(wheel)

        cv2.imshow('lane detect', wheel)

        k = cv2.waitKey(33)

        if k == 27: # esc 종료
            break
    else:
        break


