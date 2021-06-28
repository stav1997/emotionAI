import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
import pickle

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry"
# image_dir = "C:\\Users\\stav\\Desktop\\yesterday\\venv\\src\\samples\\train"

current_id = 0
file_id = 1
for root, dirs, files in os.walk(image_dir):

    for filename in files:

        if filename.endswith("png") or filename.endswith("jpg") or filename.endswith("JPG") or filename.endswith("webp"):
            path = os.path.join(root, filename)

            new_path = os.path.join(root,'a_'+str(file_id) + '_.jpg')
            os.rename(path, new_path)
            file_id = file_id + 1

