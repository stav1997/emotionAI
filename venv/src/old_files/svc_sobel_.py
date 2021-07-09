import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2 #opencv
from PIL import Image
import pickle
from skimage.feature import hog

models_dict = {'angry': 'angry_SVC_sobel_model.sav', 'disgust': 'disgust_SVC_sobel_model.sav', 'happy': 'happy_SVC_sobel_model.sav', 'natural': 'natural_SVC_sobel_model.sav', 'sad': 'sad_SVC_sobel_model.sav', 'shock': 'shock_SVC_sobel_model.sav'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
image_dir = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry"
prototxt_path = os.path.join(BASE_DIR, 'model_data\\deploy.prototxt')
caffemodel_path = os.path.join(BASE_DIR, 'model_data\\weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

data = []
names = []
# results = {}
boxes = {}
data_info = {}

def svcSobel():
    for root, dirs, files in os.walk(image_dir):
        print("svc sobel")
        for filename in files:
            results = {}
            boxes = {}
            if filename.endswith("png") or filename.endswith("jpg"):
                path = os.path.join(root, filename)
                pil_image = cv2.imread(path)

                try:

                    (h, w) = pil_image.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(pil_image, (304, 304)), 1.0, (304, 304), (104.0, 177.0, 123.0))

                    model.setInput(blob)
                    detections = model.forward()

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
                        chosen_box = boxes[min_key]
                        (startX, startY, endX, endY) = chosen_box.astype("int")
                        cv2.rectangle(pil_image, (startX, startY), (endX, endY), (255, 255, 255), 2)

                        img = Image.open(path).convert("L")
                        image_array = np.array(img, "uint8")
                        pic_ = image_array[startY:endY, startX:endX]
                        pic = cv2.resize(pic_, (304, 304))

                        dst = cv2.GaussianBlur(pic, (5, 5), cv2.BORDER_DEFAULT)
                        grad_x = cv2.Sobel(dst, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                        grad_y = cv2.Sobel(dst, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

                        abs_gard_x = cv2.convertScaleAbs(grad_x)
                        abs_gard_y = cv2.convertScaleAbs(grad_y)
                        grad = cv2.addWeighted(abs_gard_x, 0.5, abs_gard_y, 0.5, 0)

                        roi = grad.flatten()
                        roi = roi.reshape(1, -1)

                        for name, model_ in models_dict.items():
                            path_name = os.path.join(models_dir, model_)
                            pickle_in = open(path_name, 'rb')
                            model1 = pickle.load(pickle_in)
                            pickle_in.close()
                            res =model1.decision_function(roi)
                            # print(">%s --> %0.4f" % (name, res[0]))
                            results[name] = res[0]

                    except Exception as e:
                        print("1")
                        print("!!!!!!!!!!!!IMAGE " + path + " HASN'T BEEN SAVED!!!!!!!!!!!!")


                    answer = max(results, key=results.get)
                    # scale = 0.5
                    # fontScale = min(endX-startX, endY-startY) / (25 / scale)
                    # cv2.putText(pil_image, answer, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, int(fontScale), (255, 255, 0), int(2), cv2.LINE_AA)
                    results["answer"] = answer
                    print(">%s --> %s" % (filename, answer))
                    # return pil_image

                except Exception as e:
                    print("2")
                    print("!!!!!!!!!!!!IMAGE " + path+ " HASN'T BEEN SAVED!!!!!!!!!!!!")

    # print(data_info)
    return data_info
