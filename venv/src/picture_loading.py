import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
import pickle

image_dir = "C:\\Users\\stav\\Desktop\\emotionAI_\\venv\\src\\samples\\train"
# image_dir = "C:\\Users\\stav\\Desktop\\insta\\stav"

data = []
current_id = 0
file_id = 1
label_id = {}
for root, dirs, files in os.walk(image_dir):

    for filename in files:

        if filename.endswith("png") or filename.endswith("jpg"):
            path = os.path.join(root, filename)
            new_path = os.path.join(root, str(file_id) + '_.jpg')
            os.rename(path, new_path)
            file_id = file_id + 1
