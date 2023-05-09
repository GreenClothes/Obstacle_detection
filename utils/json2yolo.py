# reference : https://github.com/parksh089g/data_Pre_Processing/blob/main/json2yolo.py
# reference : https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py

import json, os, glob
from pathlib import Path

def convert_bbox(size, box):
    w = box[2] / size[0]
    h = box[3] / size[1]
    x_center = box[0] / size[0] + w / 2
    y_center = box[1] / size[1] + h / 2

    # returning yolo bbox format [x_center, y_center, w, h]
    return [x_center, y_center, w, h]

def convert_polyline(size, line):
    x_line = line[0][0::2] # extracting even index elements (x coordinate)
    y_line = line[0][1::2] # extracting odd index elements (y coordinate)

    x_min, x_max = min(x_line), max(x_line)
    y_min, y_max = min(y_line), max(y_line)

    w = (x_max - x_min) / size[0]
    h = (y_max - y_min) / size[1]
    x_center = x_min / size[0] + w / 2
    y_center = y_min / size[1] + h / 2

    # returning yolo bbox format [x_center, y_center, w, h]
    return [x_center, y_center, w, h]

def json2yolo(dir_path, save_dir):
    for json_path in dir_path:
        with open(json_path, encoding='UTF8') as jp:
            json_data = json.load(jp)

        file_name = json_path.split('\\')[-1][:-5] # get file name for ganerating .txt file

        # get image size from json_data, [width, height]
        img_size = [json_data['images'][0]['width'], json_data['images'][0]['height']]

        # get collisions of bounding boxes
        # assign a class_id using category_id
        bboxes = []
        class_id = []
        for data in json_data['annotations']:
            if 'bbox' in data:
                bbox = convert_bbox(img_size, data['bbox'])
            elif 'polyline' in data:
                bbox = convert_polyline(img_size, data['polyline'])

            bboxes.append(bbox)
            # convert category_id from 381 to 440 to class_id from 0 to 59
            if data['category_id'] == 440:
                class_id.append(data['category_id']-387)
            else:
                class_id.append(data['category_id']-381)

        # save yolo txt
        save_path = save_dir + file_name + '.txt'
        txt_file = open(save_path, 'w')
        for i in range(len(class_id)):
            label = f'{class_id[i]} {bboxes[i][0]} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]}\n'
            txt_file.write(label)
        txt_file.close()


# labeling data json files for train
train_path = glob.iglob('C:/Users/pc/Desktop/dataset/train/label_json/*.json')
train_save_dir = 'C:/Users/pc/Desktop/dataset/train/label/'

# labeling data json files for validation
val_dir_path = glob.iglob('C:/Users/pc/Desktop/dataset/val/label_json/*.json')
val_save_dir = 'C:/Users/pc/Desktop/dataset/val/label/'

# labeling data json files for test
test_dir_path = glob.iglob('C:/Users/pc/Desktop/dataset/test/label_json/*.json')
test_save_dir = 'C:/Users/pc/Desktop/dataset/test/label/'

json2yolo(train_path, train_save_dir)
json2yolo(val_dir_path, val_save_dir)
json2yolo(test_dir_path, test_save_dir)