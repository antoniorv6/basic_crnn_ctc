import os
import json
import cv2
from rich import progress
from sklearn.model_selection import train_test_split

path = 'primus2023/mono'
filepaths = []
for folder_name in progress.track(os.listdir(path)):
    folder_path = os.path.join(path, folder_name)
    for file_name in os.listdir(folder_path):
        try:
            if file_name.endswith('.json'):
                with open(os.path.join(folder_path, file_name), 'r') as json_file:
                    json_data = json.load(json_file)
                    filepaths.append((os.path.join(folder_path, file_name)))
                    
        except Exception as e:
            print('Error loading JSON file:', os.path.join(folder_path, file_name))

train, test = train_test_split(filepaths, test_size=0.25, random_state=42)
train, val = train_test_split(train, test_size=0.25, random_state=42)

with open('train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open('val.txt', 'w') as f:
    for item in val:
        f.write("%s\n" % item)

with open('test.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)

