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
pickle_in = open('pic_shock_data.pickle', 'rb')
pictures = pickle.load(pickle_in)
pickle_in.close()
print(len(pictures))

for pil_image, file_id in pictures:
    try:
        image_array = np.array(pil_image, "uint8")

        if len(pil_image.shape) >= 3:
            faces = detector.detect_faces(image_array)
            if len(faces) > 0:
                x, y, width, height = faces[0]['box']
                if y >= 0:
                    img = Image.open(file_id[1]).convert("L")  # L stands for gray scale image
                    img_gray = img.resize((480, 480))
                    image_array = np.array(img_gray, "uint8")
                    print(file_id[1])
                    # plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
                    # plt.show()
                    roi = image_array[y:y + height, x:x + width]
                    roi = cv2.resize(roi, (480, 480))

                    data.append([roi, file_id])

    except Exception as e:
        pass
print(len(data))
with open("mtcnn_shock_data.pickle", 'wb') as f:
    pickle.dump(data, f)
