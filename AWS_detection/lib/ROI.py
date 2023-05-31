import cv2, numpy as np
from param import _ROI, _img_w, _img_h, _lane_classes

def _XY2ROI(car, roadmark, img):
    ROI = [[int(_ROI[i][0]*_img_w), int(_ROI[i][1]*_img_h)] for i in range(len(_ROI))]

    roadmark_bbox = []
    lane_bbox = []

    # Dividing into lanes and roadmarks
    for i in range(len(roadmark)):
        if roadmark[i][0][0] in _lane_classes:
            lane_bbox.append(roadmark[i][1:])
        else:
            roadmark_bbox.append(roadmark[i][1:])

    # Get road lane area
    road_area = _get_road_lane(ROI, lane_bbox)
    road_area_ROI = set()

    # remove vehicle area from road lane area
    for c in car:
        for ra in range(len(road_area)):
            # if c(car bbox) is in ra(road_area), remove road_area)
            # road top pixel < car bottom pixel AND road bottom pixel > car top pixel
            # road right pixel > car left pixel AND road left pixel < car right pixel
            if not (road_area[ra][0][1] <= c[2][1] and road_area[ra][2][1] >= c[0][1] and road_area[ra][1][0] >= c[0][0] and road_area[ra][0][0] <= c[1][0]):
                road_area_ROI.add(ra)
    road_area_ROI = list(road_area_ROI).sort()
    road_area = np.copy([road_area[i] for i in road_area_ROI])

    # road_mark가 있는 차선 영역 어케 처리할지 결정
    for r in roadmark_bbox:
        for ra in range(len(road_area)):
            if road_area[ra][0][1] <= r[2][1] and road_area[ra][2][1] >= r[0][1] and road_area[ra][1][0] >= r[0][0] and road_area[ra][0][0] <= r[1][0]:
                road_area_ROI.add(ra)



    return road_area


def _get_road_lane(roi, lane):
    lane_coeff = []
    road_area = []
    road_middle = (max(roi[0]) - min(roi[0])) // 2

    for i in range(len(lane)):
        # At first, checking lanes that are in the ROI.
        # ROI top pixel < lane bottom pixel AND ROI bottom pixel > lane top pixel
        if roi[1][1] <= lane[i][2][1] and roi[4][1] > lane[i][0][1]:
            # Checking lane direction
            lane_middle = (lane[i][0][0] - lane[i][1][0]) // 2
            # Calculating coefficient of lane linear function
            if road_middle > lane_middle:
                a = np.linalg.inv([[lane[i][1][0], 1], [lane[i][3][0], 1]])
                b = [[lane[i][1][1]], [lane[i][3][1]]]
                coeff = np.dot(a, b)
                lane_coeff.append([int(coeff[0][0]), int(coeff[1][0])])
            else:
                a = np.linalg.inv([[lane[i][0][0], 1], [lane[i][2][0], 1]])
                b = [[lane[i][0][1]], [lane[i][2][1]]]
                coeff = np.dot(a, b)
                lane_coeff.append([int(coeff[0][0]), int(coeff[1][0])])

    # Divide ROI into road areas
    # Divide in order from the left
    lane_coeff = sorted(lane_coeff, key=lambda lane_coeff: lane_coeff[1])
    for lc in lane_coeff:
        road_area.append()
        #################################################################

# Removing car bboxes or road mark bboxes from road area
def _remove_area(road, remove, loc):
    if loc == 'top':
        road[0][1] = remove[2][1]
        road[1][1] = remove[2][1]
    elif loc == 'bottom':
        road[2][1] = remove[0][1]
        road[3][1] = remove[0][1]
    elif loc == 'middle':
        top = remove[0][1] - road[0][1]
        bottom = road[2][1] - remove[2][1]
        if top >= bottom:
            road[2][1] = remove[0][1]
            road[3][1] = remove[0][1]
        else:
            road[0][1] = remove[2][1]
            road[1][1] = remove[2][1]