import cv2, numpy as np

img = cv2.imread('images/traffic.png')

def ROI(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    '''
    region = np.array([[0, img_h * 0.76], [img_w * 0.31, img_h * 0.58], [img_w * 0.79, img_h * 0.58],
                       [img_w, img_h * 0.72], [img_w, img_h*0.9], [0, img_h*0.9]], np.int32)
    '''
    region = np.array([[img_w * 0.31, img_h * 0.6], [img_w * 0.79, img_h * 0.6],
                       [img_w, img_h * 0.9], [0, img_h * 0.9]], np.int32)
    mask = np.zeros((img_h, img_w, 3), np.uint8)
    cv2.fillPoly(mask, [region], (255, 255, 255))
    #cv2.imshow('mask', mask)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

mimg = ROI(img)
cv2.imshow('ming', mimg)
cv2.waitKey(0)
