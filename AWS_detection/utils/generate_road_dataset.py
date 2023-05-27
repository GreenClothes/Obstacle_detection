import cv2, glob
import numpy as np

img_num = 0
save_dir = 'C:/Users/pc/Desktop/road/512/'

def PERSPECTIVE_TRANS(img, xy):
    img_h = img.shape[0]
    img_w = img.shape[1]

    before_trans = np.float32([xy[0], xy[1], xy[2], xy[3]])
    after_trans = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])

    trans_mtrx = cv2.getPerspectiveTransform(before_trans, after_trans)
    trans_img = cv2.warpPerspective(img, trans_mtrx, (512, 512))

    return trans_img

def onMouse(event, x, y, flags, param):
    global img_num
    if event == cv2.EVENT_LBUTTONDOWN and len(click_point) != 4:
        click_point.append([x, y])
    if len(click_point) == 1:
        cv2.circle(img, (click_point[0][0], click_point[0][1]), 3, (255, 0, 0), -1)
    if len(click_point) >= 2:
        cv2.line(img, (click_point[-2][0], click_point[-2][1]), (click_point[-1][0], click_point[-1][1]), (255, 0, 0), 5)
    if len(click_point) == 4:
        cv2.line(img, (click_point[0][0], click_point[0][1]), (click_point[-1][0], click_point[-1][1]), (255, 0, 0), 5)
        # convert the specified area to a 512x512 image
        save_path = f'{save_dir}road{img_num}.jpg'
        cv2.imwrite(save_path, PERSPECTIVE_TRANS(image, click_point))
        img_num = img_num + 1
        click_point.clear()
    cv2.imshow('img', img)

image_dir = glob.iglob('C:/Users/pc/Desktop/road/normal/*.jpg')

for image in image_dir:
    image = cv2.resize(cv2.imread(image), (1920, 1080))
    img = np.copy(image)
    cv2.imshow('img', img)
    while 1:
        click_point = []
        cv2.setMouseCallback('img', onMouse)
        keycode = cv2.waitKey()

        if keycode == 13:
            click_point.clear()
        if keycode == ord('n'): # push 'n' for next image
            break
cv2.destroyAllWindows()

# image creation with brightness control and blurring
saved_img_dir = f'{save_dir}*.jpg'
saved_img_dir = glob.iglob(saved_img_dir)

for img in saved_img_dir:
    img512 = cv2.imread(img)
    img_name = img.split('\\')[1][:-4]

    for i in range(20):
        bright_array = np.full((512, 512, 3), (i*3, i*3, i*3), dtype=np.uint8)

        add = cv2.add(img512, bright_array)
        sub = cv2.subtract(img512, bright_array)
        blur = cv2.blur(img512, (i+1, i+1), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

        cv2.imwrite(save_dir + img_name + '_' + str(3 * i) + '.jpg', add)
        cv2.imwrite(save_dir + img_name + '_' + str(3 * i + 1) + '.jpg', sub)
        cv2.imwrite(save_dir + img_name + '_' + str(3 * i + 2) + '.jpg', blur)
