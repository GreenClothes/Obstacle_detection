import numpy as np, cv2


def Center_line_offset(img, l_pt, r_pt, inv_mtrx):
    color_perspect = np.zeros_like(img).astype(np.uint8)

    midpoint_detect = (r_pt[0, 0, 0] + l_pt[-1, -1, -2])//2
    midpoint_img = img.shape[1] // 2
    offset = midpoint_img - midpoint_detect

    l_pt[-1, -1::-1, -2] = (r_pt[0, :, 0] + l_pt[-1, -1::-1, -2])//2
    r_pt[0, :, 0] = l_pt[-1, -1::-1, -2] + offset

    pts = np.hstack((l_pt, r_pt))

    if abs(offset) < 30: color = (0, 255, 0)
    else: color = (0, 0, 255)

    color_wind = cv2.fillPoly(color_perspect, np.int_([pts]), color)
    wind_perspect = cv2.warpPerspective(color_wind, inv_mtrx, (img.shape[1], img.shape[0]))
    offset_img = cv2.addWeighted(img, 1, wind_perspect, 0.5, 0)

    text = "{0}cm offset".format(offset*0.5)
    org = (img.shape[1]*3//7, img.shape[0]//10)

    if abs(offset) < 30:
        offset_img = cv2.putText(offset_img, text, org,
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        offset_img = cv2.putText(offset_img, text, org,
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        offset_img = cv2.putText(offset_img, 'CAUTION', (img.shape[1]*3//7, img.shape[0]//6),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    return offset_img, l_pt

def rotate(img, degree):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated

def steering_wheel(img, overlap_img, mid_pt):
    degree = np.arctan((mid_pt[-1, -1, 0] - mid_pt[0, 0, 0])/mid_pt[-1, -1, 1]*np.pi)
    overlap_img = rotate(overlap_img, degree*20)
    overlap_img = overlap_img[overlap_img.shape[1]//2-115:overlap_img.shape[1]//2+115,
                              overlap_img.shape[0]//2-115:overlap_img.shape[0]//2+120]
    overlap_img_gray = cv2.cvtColor(overlap_img, cv2.COLOR_BGR2GRAY)

    rows, cols, _ = overlap_img.shape
    roi = img[10:rows+10, 10:cols+10]

    ret, masks = cv2.threshold(overlap_img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    inv_masks = cv2.bitwise_not(masks)

    overlap_img_bg = cv2.bitwise_and(roi, roi, mask = inv_masks)
    overlap_img_fg = cv2.bitwise_and(overlap_img, overlap_img, mask = masks)

    dst = cv2.add(overlap_img_bg, overlap_img_fg)
    img[10:rows+10, 10:cols+10] = dst

    text = "{0:0.2f}degree".format(degree * 20)
    org = ((rows+10)//2-60, cols+30)
    img = cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return img