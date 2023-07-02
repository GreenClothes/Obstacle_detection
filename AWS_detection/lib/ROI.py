import cv2, numpy as np
from lib.param import _img_w, _img_h, _lane_classes, _source_img_dir, _ROI

def _XY2ROI(car, roadmark):

    roadmark_bbox = []
    lane_bbox = []

    # Dividing into lanes and roadmarks
    for i in range(len(roadmark)):
        if roadmark[i][0][0] in _lane_classes:
            lane_bbox.append(roadmark[i][1:])
        else:
            roadmark_bbox.append(roadmark[i][1:])

    # Get ROI(road area) from img
    lane_bbox = sorted(lane_bbox, key=lambda lane_bbox:lane_bbox[1][0])


    # Get road lane area
    road_area, lane_coeff = _get_road_lane(_ROI, lane_bbox)

    # remove vehicle area from road lane area
    for c in car:
        for ra in range(len(road_area)):
            # if c(car bbox) is in ra(road_area), remove c from ra
            # road top pixel < car bottom pixel AND road bottom pixel > car top pixel
            if road_area[ra][0][1] <= c[2][1] and road_area[ra][2][1] >= c[0][1]:
                _remove_area(road_area, ra, c, _in_ROI(road_area[ra], ra, c, lane_coeff), lane_coeff)

    # remove road mark area from road lane area
    for r in roadmark_bbox:
        for ra in range(len(road_area)):
            # if r(road mark bbox) is in ra(road_area), remove r from ra
            # road top pixel < road mark bottom pixel AND road bottom pixel > road mark top pixel
            if road_area[ra][0][1] <= r[2][1] and road_area[ra][2][1] >= r[0][1]:
                _remove_area(road_area, ra, c, _in_ROI(road_area[ra], ra, c, lane_coeff), lane_coeff)

    return road_area


def _get_road_lane(roi, lane):
    lane_coeff = []
    road_area = []
    road_middle = (roi[2][0] + roi[3][0]) // 2
    for i in range(len(lane)):
        # At first, checking lanes that are in the ROI.
        # ROI top pixel < lane bottom pixel AND ROI bottom pixel > lane top pixel
        if (roi[0][1] <= lane[i][2][1] and roi[2][1] >= lane[i][0][1]) or (roi[0][1] >= lane[i][2][1] and roi[2][1] <= lane[i][0][1]):
            # Checking lane direction
            lane_middle = (lane[i][1][0] + lane[i][0][0]) // 2
            # Calculating coefficient of lane linear function
            if road_middle > lane_middle:
                a = np.linalg.inv([[lane[i][1][0], 1], [lane[i][3][0], 1]])
                b = [[lane[i][1][1]], [lane[i][3][1]]]
                coeff = np.dot(a, b)
                lane_coeff.append([coeff[0][0], coeff[1][0]])
            else:
                a = np.linalg.inv([[lane[i][0][0], 1], [lane[i][2][0], 1]])
                b = [[lane[i][0][1]], [lane[i][2][1]]]
                coeff = np.dot(a, b)
                lane_coeff.append([coeff[0][0], coeff[1][0]])

    # Divide ROI into road areas
    # Divide in order from the left
    lane_coeff = sorted(lane_coeff, key=lambda lane_coeff: lane_coeff[1]/lane_coeff[0])
    for lc in range(len(lane_coeff)-1):
        topLeft = [(roi[0][1]-lane_coeff[lc][1])//lane_coeff[lc][0]+_img_w*0.01, roi[0][1]]
        topRight = [(roi[0][1]-lane_coeff[lc+1][1])//lane_coeff[lc+1][0]-_img_w*0.01, roi[0][1]]
        bottomRight = [(roi[2][1]-lane_coeff[lc+1][1])//lane_coeff[lc+1][0]-_img_w*0.01, roi[2][1]]
        bottomLeft = [(roi[2][1]-lane_coeff[lc][1])//lane_coeff[lc][0]+_img_w*0.01, roi[2][1]]
        if abs(topRight[0] - topLeft[0]) >= _img_w*0.05 or abs(bottomRight[0] - bottomLeft[0]) > _img_w*0.05:
            road_area.append([topLeft, topRight, bottomRight, bottomLeft])

    return road_area, lane_coeff

# Removing car bboxes or road mark bboxes from road area
def _remove_area(road, idx, remove, loc, coeff):
    if loc == 'top':
        xLeft = int((remove[2][1] - coeff[idx][1])//coeff[idx][0])
        xRight = int((remove[2][1] - coeff[idx+1][1])//coeff[idx+1][0])
        road[idx][0][0] = xLeft+_img_w*0.001
        road[idx][0][1] = remove[2][1]
        road[idx][1][0] = xRight
        road[idx][1][1] = remove[2][1]
    elif loc == 'bottom':
        xLeft = int((remove[0][1] - coeff[idx][1]) // coeff[idx][0])
        xRight = int((remove[0][1] - coeff[idx + 1][1]) // coeff[idx + 1][0])
        road[idx][2][0] = xRight-_img_w*0.01
        road[idx][2][1] = remove[0][1]
        road[idx][3][1] = xLeft+_img_w*0.01
        road[idx][3][1] = remove[0][1]
    elif loc == 'middle':
        top = remove[0][1] - road[idx][0][1]
        bottom = road[idx][2][1] - remove[2][1]
        if top >= bottom:
            xLeft = int((remove[0][1] - coeff[idx][1]) // coeff[idx][0])
            xRight = int((remove[0][1] - coeff[idx + 1][1]) // coeff[idx + 1][0])
            road[idx][2][0] = xRight-_img_w*0.01
            road[idx][2][1] = remove[0][1]
            road[idx][3][1] = xLeft+_img_w*0.01
            road[idx][3][1] = remove[0][1]
        else:
            xLeft = int((remove[2][1] - coeff[idx][1]) // coeff[idx][0])
            xRight = int((remove[2][1] - coeff[idx + 1][1]) // coeff[idx + 1][0])
            road[idx][0][0] = xLeft+_img_w*0.01
            road[idx][0][1] = remove[2][1]
            road[idx][1][0] = xRight-_img_w*0.01
            road[idx][1][1] = remove[2][1]

    return road

# Checking objects(cars or road marks) are in road area
def _in_ROI(road, idx, obj, coeff):
    if road[0][1] <= obj[2][1] and road[2][1] >= obj[0][1]:
        obj_topLeft = (obj[0][1] - coeff[idx][1]) // coeff[idx][0]
        obj_topRight = (obj[0][1] - coeff[idx+1][1]) // coeff[idx+1][0]
        obj_bottomRight = (obj[2][1] - coeff[idx+1][1]) // coeff[idx+1][0]
        obj_bottomLeft = (obj[2][1] - coeff[idx][1]) // coeff[idx][0]

        if (obj[0][0] <= obj_topRight and obj[1][0] >= obj_topLeft) or (obj[3][0] <= obj_bottomRight and obj[2][0] >= obj_bottomLeft):
            if road[0][1] >= obj[0][1]:
                return 'top'
            elif road[2][1] <= obj[2][1]:
                return 'bottom'
            else:
                return 'middle'
