import numpy as np
import cv2
import os

img_path = os.getcwd()+'\\yolov5\\data\\images\\detect\\test.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_h = img.shape[0]
img_w = img.shape[1]

region = np.array([[0, int(img_h*0.45)], [img_w, int(img_h*0.45)], [img_w, int(img_h*0.65)], [0, int(img_h*0.65)]])

mask = np.zeros((img_h, img_w, 3), np.uint8)
cv2.fillPoly(mask, [region], (255, 255, 255))
print(img.shape)
print(mask.shape)
masked_img = cv2.bitwise_and(img, mask)

cv2.imshow('img', masked_img)
cv2.waitKey(0)