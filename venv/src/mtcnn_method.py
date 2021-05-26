import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2 #opencv
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
detector = MTCNN()

data = []
pictures = []
pickle_in = open('pic_data.pickle', 'rb')
pictures = pickle.load(pickle_in)
pickle_in.close()
for pil_image, file_id in pictures:
    try:
        img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
        image_array = np.array(img, "uint8")
        if len(img.shape) >= 3:
            faces = detector.detect_faces(image_array)
            if len(faces) > 0:
                x, y, width, height = faces[0]['box']
                if y >= 0:
                    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    image_array = np.array(image, "uint8")
                    roi = image_array[y:y + height, x:x + width]
                    roi = cv2.resize(roi, (480, 480))
                    data.append([roi, file_id])

    except Exception as e:
        pass

with open("mtcnn_data.pickle", 'wb') as f:
    pickle.dump(data, f)
