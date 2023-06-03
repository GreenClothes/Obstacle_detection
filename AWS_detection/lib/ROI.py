import cv2, numpy as np
from lib.param import _img_w, _img_h, _lane_classes

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
    ROI = [[lane_bbox[0][1][0], lane_bbox[0][1][1]], [lane_bbox[-1][0][0], lane_bbox[-1][0][1]],
           [lane_bbox[-1][2][0], lane_bbox[-1][2][1]], [lane_bbox[0][3][0], lane_bbox[0][3][1]]]
    #ROI = [[int(ROI[i][0] * _img_w), int(ROI[i][1] * _img_h)] for i in range(len(ROI))]

    # Get road lane area
    road_area = _get_road_lane(ROI, lane_bbox)

    # remove vehicle area from road lane area
    for c in car:
        for ra in range(len(road_area)):
            # if c(car bbox) is in ra(road_area), remove c from road area)
            # road top pixel < car bottom pixel AND road bottom pixel > car top pixel
            # road right pixel > car left pixel AND road left pixel < car right pixel
            if road_area[ra][0][1] <= c[2][1] and road_area[ra][2][1] >= c[0][1] and road_area[ra][1][0] >= c[0][0] and road_area[ra][0][0] <= c[1][0]:
                # top
                if road_area[ra][0][1] >= c[0][1]:
                    _remove_area(road_area[ra], c, 'top')
                # bottom
                elif road_area[ra][2][1] <= c[2][1]:
                    _remove_area(road_area[ra], c, 'bottom')
                # middle
                else:
                    _remove_area(road_area[ra], c, 'middle')

    # remove road mark area from road lane area
    for r in roadmark_bbox:
        for ra in range(len(road_area)):
            if road_area[ra][0][1] <= r[2][1] and road_area[ra][2][1] >= r[0][1] and road_area[ra][1][0] >= r[0][0] and road_area[ra][0][0] <= r[1][0]:
                # top
                if road_area[ra][0][1] >= r[0][1]:
                    _remove_area(road_area[ra], r, 'top')
                # bottom
                elif road_area[ra][2][1] <= r[2][1]:
                    _remove_area(road_area[ra], r, 'bottom')
                # middle
                else:
                    _remove_area(road_area[ra], r, 'middle')

    return road_area


def _get_road_lane(roi, lane):
    lane_coeff = []
    road_area = []
    road_middle = (roi[2][0] + roi[3][0]) // 2
    for i in range(len(lane)):
        # At first, checking lanes that are in the ROI.
        # ROI top pixel < lane bottom pixel AND ROI bottom pixel > lane top pixel
        if roi[0][1] <= lane[i][2][1] or roi[2][1] >= lane[i][0][1]:
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

    #########################################################
    # opencv로 lane_coeff에 따라서 이미지 위에 직선 그려보기
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → img
    # cv2.line(black_canvas, (10, 100), (500, 100), green)
    #########################################################

    # Divide ROI into road areas
    # Divide in order from the left
    lane_coeff = sorted(lane_coeff, key=lambda lane_coeff: lane_coeff[1])
    for lc in range(len(lane_coeff)-1):
        road_area.append([[roi[0][1]//lane_coeff[lc][0]-lane_coeff[lc][1]+_img_w*0.01, roi[0][0]*lane_coeff[lc][0]+lane_coeff[lc][1]],
                          [roi[0][1]//lane_coeff[lc+1][0]-lane_coeff[lc+1][1]-_img_w*0.01, roi[0][0]*lane_coeff[lc+1][0]+lane_coeff[lc+1][1]],
                          [roi[2][1]//lane_coeff[lc+1][0]-lane_coeff[lc+1][1]-_img_w*0.01, roi[2][1]*lane_coeff[lc+1][0]+lane_coeff[lc+1][1]],
                          [roi[2][1]//lane_coeff[lc][0]-lane_coeff[lc+1][1]+_img_w*0.01, roi[2][1]*lane_coeff[lc][0]+lane_coeff[lc][1]]])

    return road_area

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