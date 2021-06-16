import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples\\train")
data = []
current_id = 0
file_id = 1000
label_id = {}
categories = []
for root, dirs, files in os.walk(image_dir):

    for filename in files:
        if filename.endswith("png") or filename.endswith("jpg"):
            path = os.path.join(root, filename)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_id:
                categories.append(label)
                label_id[label] = current_id
                current_id += 1

            id_ = label_id[label]
        # if filename.endswith("png") or filename.endswith("jpg"):
        #     path = os.path.join(root, filename)
        #     new_path = os.path.join(root, '_'+str(file_id) + '_.jpg')
        #     os.rename(path, new_path)
        #     file_id = file_id + 1
        #     label = os.path.basename(os.path.dirname(new_path)).replace(" ", "-").lower()
        #     # label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
        #
        #     if not label in label_id:
        #         label_id[label] = current_id
        #         current_id += 1
        #
        #     id_ = label_id[label]
            pil_image = cv2.imread(path)

            try:
                img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
                file_id = [label, path]
                data.append([img, file_id])

            except Exception as e:
                pass


with open('pic_shock_data.pickle', 'wb') as f:
    pickle.dump(data, f)

