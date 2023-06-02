import numpy as np, cv2

def _Read_coordinate(file_path, img_h, img_w, road_mark=False):
    #img_h = img.shape[0]
    #img_w = img.shape[1]
    #print('img_h :', img_h, 'img_w :', img_w)

    try:
        f = open(file_path, 'r')
        #print('file!')
    except:
        #print('no file')
        return

    coordinate_list = []

    while True:
        line = list(map(float, f.readline().split()))
        if not line: break

        line[1] = int(line[1] * img_w)
        line[2] = int(line[2] * img_h)
        line[3] = int(line[3] * img_w)
        line[4] = int(line[4] * img_h)

        # line[0] : class, 0 : dummy data for dimension
        # Sequence : aleft upper, right upper, right lower, left lower
        coordinate = np.array([[line[0], 0],
                               [line[1] - line[3]//2, line[2] - line[4]//2],
                               [line[1] + line[3]//2, line[2] - line[4]//2],
                               [line[1] + line[3]//2, line[2] + line[4]//2],
                               [line[1] - line[3]//2, line[2] + line[4]//2]], np.int32)
        coordinate_list.append(coordinate)
    f.close()

    #print(*coordinate_list)
    return coordinate_list