import numpy as np, cv2

#file_path = 'C:\\Users\\kkb99\\Desktop\\detection\\yolov5\\runs\\detect\\exp3\\labels\\trans_img.txt'

def read_coordinate(file_path, img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    #print('img_h :', img_h, 'img_w :', img_w)

    f = open(file_path, 'r')
    coordinate_list = []

    while True:
        line = list(map(float, f.readline().split()))
        if not line: break
        del line[0]
        '''
        coordinate[0] = int(line[0] * img_w)
        coordinate[1] = int(line[1] * img_h)
        coordinate[2] = int(line[2] * img_w)
        coordinate[3] = int(line[3] * img_h)
        '''

        line[0] = int(line[0] * img_w)
        line[1] = int(line[1] * img_h)
        line[2] = int(line[2] * img_w)
        line[3] = int(line[3] * img_h)

        coordinate = np.array([[line[0] - line[2]//2, line[1] - line[3]//2], [line[0] + line[2]//2, line[1] - line[3]//2],
                              [line[0] + line[2]//2, line[1] + line[3]//2], [line[0] - line[2]//2, line[1] + line[3]//2]],
                              np.int32)
        coordinate_list.append(coordinate)
    f.close()

    #print(*coordinate_list)
    return coordinate_list