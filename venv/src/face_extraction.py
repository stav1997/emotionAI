import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2 #opencv
from PIL import Image
import pickle
from face_extractor import Extractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples\\train")
new_image_dir = os.path.join(BASE_DIR, "samples\\updated_images")
roi_image_dir = os.path.join(BASE_DIR, "samples\\faces")
print(BASE_DIR)

data = []
current_id = 0
file_id = 0
label_id = {}
categories = []


base_dir = os.path.dirname(__file__)
print(base_dir)
prototxt_path = os.path.join(BASE_DIR, 'model_data\\deploy.prototxt')
caffemodel_path = os.path.join(BASE_DIR, 'model_data\\weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
data = []
for root, dirs, files in os.walk(image_dir):

    for filename in files:
        boxes ={}
        if filename.endswith("png") or filename.endswith("jpg"):

            path = os.path.join(root, filename)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_id:
                categories.append(label)
                label_id[label] = current_id
                current_id += 1

            id_ = label_id[label]
            pil_image = cv2.imread(path)
            # image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
            try:
                (h, w) = pil_image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(pil_image, (304, 304)), 1.0, (304, 304), (104.0, 177.0, 123.0))

                model.setInput(blob)
                detections = model.forward()
                # print("starting image " + filename)
                for i in range(0, detections.shape[2]):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.9:
                        boxes[startX] = box
                    else:
                        break
                try:
                    min_key = min(boxes, key=float)
                    # print(boxes)
                    # print("min key is: ",min_key)
                    chosen_box = boxes[min_key]
                    (startX, startY, endX, endY) = chosen_box.astype("int")
                    frame = pil_image[startY:endY, startX:endX]
                    roi = cv2.resize(frame, (304, 304))
                    cv2.imwrite(roi_image_dir + '\\' + label + '\\' + filename, roi)
                    cv2.rectangle(pil_image, (startX, startY), (endX, endY), (255, 255, 255), 2)
                    cv2.imwrite(new_image_dir + '\\' + label + '\\' + filename, pil_image)
                    img = Image.open(path).convert("L")
                    image_array = np.array(img, "uint8")
                    pic_ = image_array[startY:endY, startX:endX]
                    pic = cv2.resize(pic_, (304, 304))
                    data.append([pic, label])
                except Exception as e:
                    print("!!!!!!!!!!!!IMAGE " + filename + " HASN'T BEEN SAVED!!!!!!!!!!!!")
                #
                # print("Image " + filename + " converted successfully")
                # print("Image " + filename + " ROI successfully")

            except Exception as e:
                pass



with open('pics.pickle', 'wb') as f:
    pickle.dump(data, f)
