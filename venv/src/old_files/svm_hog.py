import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2  # opencv
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
from skimage.feature import hog

detector = MTCNN()
accuracy = {}
with open("C:\\Users\\stav\\Desktop\\emotionAI\\venv\\src\\disgust_model.sav", 'rb') as f:
    model = pickle.load(f)
    f.close()
    
def svmHog(path):
    pil_image = cv2.imread(path)

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
                    fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    roi = hog_image.flatten()
                    roi = roi.reshape(1, -1)
                    pred = model.predict(roi)
                    if(pred[0] == 1):
                        print("disgust!")
                    else:
                        print("not disgust!")

    except Exception as e:
        pass

    return image_array