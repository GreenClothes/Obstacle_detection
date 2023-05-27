import numpy as np, cv2

def ROI(img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    region = np.array([[img_w * 0.44, img_h * 0.64], [img_w * 0.51, img_h * 0.64], [img_w, img_h], [0, img_h]], np.int32)

    mask = np.zeros((img_h, img_w), np.uint8)
    cv2.fillPoly(mask, [region], 255)
    #cv2.imshow('mask', mask)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def PERSPECTIVE_TRANS(img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    before_trans = np.float32([[img_w * 0.44, img_h * 0.64], [img_w * 0.51, img_h * 0.64], [img_w, img_h], [0, img_h]])
    after_trans = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    inverse_mtrx = cv2.getPerspectiveTransform(after_trans, before_trans)
    trans_mtrx = cv2.getPerspectiveTransform(before_trans, after_trans)
    trans_img = cv2.warpPerspective(img, trans_mtrx, (img.shape[1], img.shape[0]))

    return trans_img, inverse_mtrx