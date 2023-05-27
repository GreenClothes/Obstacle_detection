import os

_source_img_dir = 'C:\\Users\\pc\\Desktop\\dataset\\yolo_src\\brick.png'
_source_car_weights_dir = os.getcwd()+'\\yolov5\\weights\\yolov5s.pt'
_source_roadmark_weights_dir = os.getcwd()+'\\yolov5\\weights\\best.pt'
_save_txt = True
_nosave = False
_classes = [2, 3, 5, 7] # coco128.yaml (2 : car / 3 : motorcycle / 5 : bus / 7 : truck)
_save_car_name = 'car'
_save_road_mark_name = 'roadmark'
_detect_result_dir = '\\yolov5\\runs\\detect\\'
_img_h = 1080
_img_w = 1920
_ROI = [[0, 0.86], [0.31, 0.68], [0.79, 0.68], [1, 0.82], [1, 1], [0, 1]]
_lane_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # road lane classes