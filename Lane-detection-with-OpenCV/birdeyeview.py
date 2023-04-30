import numpy as np, cv2

def ROI(img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    region = np.array([[img_w * 0.35, img_h * 0.3], [img_w * 0.48, img_h * 0.3], [img_w, img_h], [0, img_h]], np.int32)

    mask = np.zeros((img_h, img_w, 3), np.uint8)
    cv2.fillPoly(mask, [region], (255, 255, 255))
    #cv2.imshow('mask', mask)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def PERSPECTIVE_TRANS(img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    before_trans = np.float32([[img_w * 0.35, img_h * 0.3], [img_w * 0.48, img_h * 0.3], [img_w, img_h], [0, img_h]])
    after_trans = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    inverse_mtrx = cv2.getPerspectiveTransform(after_trans, before_trans)
    trans_mtrx = cv2.getPerspectiveTransform(before_trans, after_trans)
    trans_img = cv2.warpPerspective(img, trans_mtrx, (img.shape[1], img.shape[0]))

    return trans_img, inverse_mtrx

img = cv2.imread('images/road_detect(2).jpg')
if img is None: raise Exception('No image')

ROI_img = ROI(img)

#cv2.imshow('ROI img', ROI_img)

PT_img, inverse_mtrx = PERSPECTIVE_TRANS(ROI_img)

#cv2.imshow('PT img', PT_img)
#cv2.waitKey(0)
#cv2.imwrite('images/trans_img.jpg', PT_img)

img_h = PT_img.shape[0]
img_w = PT_img.shape[1]

region = []
region.append(np.array([[0, 0], [img_w * 0.15, 0], [img_w * 0.15, img_h], [0, img_h]], np.int32))
region.append(np.array([[img_w * 0.23, 0], [img_w * 0.55, 0], [img_w * 0.55, img_h], [img_w * 0.23, img_h]], np.int32))
region.append(np.array([[img_w * 0.7, 0], [img_w, 0], [img_w, img_h], [img_w * 0.7, img_h]], np.int32))

mask = np.zeros((img_h, img_w, 3), np.uint8)
for i in range(len(region)):
    cv2.fillPoly(mask, [region[i]], (255, 255, 255))
masked_img = cv2.bitwise_and(PT_img, mask)

#cv2.imshow('mask img', masked_img)
#cv2.waitKey(0)

from read_txt import read_coordinate

file_path = 'C:\\Users\\pc\\Desktop\\Obstacle_detection\\yolov5\\runs\\detect\\exp2\\labels\\trans_img.txt'
bbox = read_coordinate(file_path, masked_img)

#print(*bbox)

mask = np.full((img_h, img_w, 3), (255, 255, 255), np.uint8)
for i in range(len(bbox)):
    cv2.fillPoly(mask, [bbox[i]], (0, 0, 0))
#cv2.imshow('mask', mask)
masked_img = cv2.bitwise_and(masked_img, mask)

#print(masked_img)
cv2.imshow('img', PT_img)
cv2.imshow('after yolo', masked_img)
cv2.waitKey(0)
cv2.imwrite('C:\\Users\\pc\\Desktop\\PT_img.jpg', PT_img)
cv2.imwrite('C:\\Users\\pc\\Desktop\\masked_img.jpg', masked_img)