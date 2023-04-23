import numpy as np, cv2

def sliding_window(img):

    nwindow = 5
    margin = 100
    minpix = 50
    lane_bin_th = 145

    _, lane = cv2.threshold(img, lane_bin_th, 255, cv2.THRESH_BINARY)

    histogram = np.sum(lane[lane.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    left_max = np.argmax(histogram[:midpoint])
    right_max = np.argmax(histogram[midpoint:]) + midpoint

    wind_h = np.int32(lane.shape[0]/nwindow)
    nz_index = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx, ly, rx, ry = [], [], [], []

    output = np.dstack((lane, lane, lane)) * 255

    for window in range(nwindow):
        win_yl = lane.shape[0] - (window + 1) * wind_h
        win_yh = lane.shape[0] - window * wind_h

        win_xll = left_max - margin
        win_xlh = left_max + margin
        win_xrl = right_max - margin
        win_xrh = right_max + margin

        cv2.rectangle(output, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        cv2.rectangle(output, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

        detect_left_inds = ((nz_index[0] >= win_yl) & (nz_index[0] < win_yh) & (nz_index[1] >= win_xll) & (nz_index[1] < win_xlh)).nonzero()[0]
        detect_right_inds = ((nz_index[0] >= win_yl) & (nz_index[0] < win_yh) & (nz_index[1] >= win_xrl) & (nz_index[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(detect_left_inds)
        right_lane_inds.append(detect_right_inds)

        if len(detect_left_inds) > minpix:
            left_max = np.int32(np.mean(nz_index[1][detect_left_inds]))
        if len(detect_right_inds) > minpix:
            right_max = np.int32(np.mean(nz_index[1][detect_right_inds]))

        lx.append(left_max)
        ly.append((win_yl + win_yh)/2)

        rx.append(right_max)
        ry.append(((win_yl + win_yh)/2))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    output[nz_index[0][left_lane_inds], nz_index[1][left_lane_inds]] = [255, 0, 0]
    output[nz_index[0][right_lane_inds], nz_index[1][right_lane_inds]] = [0, 0, 255]
    #cv2.imshow('test', output)

    return lfit, rfit

def draw_window(img, perspect_img, inverse_mtrx, lfit, rfit):
    y_max = perspect_img.shape[0]
    y_plot = np.linspace(0, y_max - 1, y_max)
    color_perspect = np.zeros_like(img).astype(np.uint8)

    lfit_x = lfit[0] * y_plot**2 + lfit[1] * y_plot + lfit[2]
    rfit_x = rfit[0] * y_plot**2 + rfit[1] * y_plot + rfit[2]

    l_pts = np.array([np.transpose(np.vstack([lfit_x, y_plot]))])
    r_pts = np.array([np.flipud(np.transpose(np.vstack([rfit_x, y_plot])))])
    pts = np.hstack((l_pts, r_pts))

    color_wind = cv2.fillPoly(color_perspect, np.int_([pts]), (255, 0, 0))
    wind_perspect = cv2.warpPerspective(color_wind, inverse_mtrx, (img.shape[1], img.shape[0]))

    return cv2.addWeighted(img, 1, wind_perspect, 0.3, 0), l_pts, r_pts