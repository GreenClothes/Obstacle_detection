import numpy as np
import cv2

img_path = 'C:\\Users\\pc\\Desktop\\Obstacle_detection\\AWS_detection\\yolov5\\data\\images\\test\\road(11).jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

img_h = img.shape[0]
img_w = img.shape[1]

region = np.array([[img_w*0.3, img_h * 0.3], [img_w*0.7, img_h * 0.3],
                    [img_w, img_h * 0.7], [0, img_h * 0.7]], np.int32)
mask = np.zeros((img_h, img_w, 3), np.uint8)
cv2.fillPoly(mask, [region], (255, 255, 255))
masked_img = cv2.bitwise_and(img, mask)
#cv2.imshow('img', masked_img)
#cv2.waitKey(0)

region = np.array([[0, img_h * 0.3], [img_w, img_h * 0.3],
                    [img_w, img_h * 0.7], [0, img_h * 0.7]], np.float32)

b_trans = np.copy(region)
a_trans = np.float32([[img_w*0.3, img_h*0.5], [img_w*0.7, img_h*0.5], [img_w, img_h], [0, img_h]])

trans_mtrx = cv2.getPerspectiveTransform(b_trans, a_trans)
trans_img = cv2.warpPerspective(img, trans_mtrx, (img_w, img_h))

cv2.imshow('img', trans_img)
cv2.waitKey(0)
cv2.imwrite('C:\\Users\\pc\\Desktop\\trans_img.jpg', trans_img)