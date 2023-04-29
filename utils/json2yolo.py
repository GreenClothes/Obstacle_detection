# reference : https://github.com/parksh089g/data_Pre_Processing/blob/main/json2yolo.py

import json, os, glob
from pathlib import Path

val_dir = glob.iglob('C:\\Users\\pc\\Desktop\\dataset\\val\\label_json\\*.json')
val_save_dir = 'C:\\Users\\pc\\Desktop\\dataset\\val\\label'

for json_file in val_dir:
    with open (json_file) as jf:
        json_data = json.load(jf)

    # get file name
    file_name = ''

    for d in json_data['annotations']:
        