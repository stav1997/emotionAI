import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2
from PIL import Image
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
prototxt_path = os.path.join(BASE_DIR, 'model_data\\deploy.prototxt')
caffemodel_path = os.path.join(BASE_DIR, 'model_data\\weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

image_dir = os.path.join(BASE_DIR, "samples\\validation")

OCS_sobel_models_dict = {'angry': 'angry_OCS_sobel_model.sav', 'disgust': 'disgust_OCS_sobel_model.sav',
                         'happy': 'happy_OCS_sobel_model.sav', 'natural': 'natural_OCS_sobel_model.sav',
                         'sad': 'sad_OCS_sobel_model.sav', 'shock': 'shock_OCS_sobel_model.sav'}
OCS_hog_models_dict = {'angry': 'angry_OCS_hog_model.sav', 'disgust': 'disgust_OCS_hog_model.sav',
                       'happy': 'happy_OCS_hog_model.sav', 'natural': 'natural_OCS_hog_model.sav',
                       'sad': 'sad_OCS_hog_model.sav', 'shock': 'shock_OCS_hog_model.sav'}
SVC_hog_models_dict = {'angry': 'angry_SVC_hog_model.sav', 'disgust': 'disgust_SVC_hog_model.sav',
                       'happy': 'happy_SVC_hog_model.sav', 'natural': 'natural_SVC_hog_model.sav',
                       'sad': 'sad_SVC_hog_model.sav', 'shock': 'shock_SVC_hog_model.sav'}
SVC_sobel_models_dict = {'angry': 'angry_SVC_sobel_model.sav', 'disgust': 'disgust_SVC_sobel_model.sav',
                         'happy': 'happy_SVC_sobel_model.sav', 'natural': 'natural_SVC_sobel_model.sav',
                         'sad': 'sad_SVC_sobel_model.sav', 'shock': 'shock_SVC_sobel_model.sav'}


def readSobel(path_, models, many=None):
    results = {}
    boxes = {}
    models_dict = models
    pil_image = cv2.imread(path_)

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

            img = Image.open(path_).convert("L")
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

                res = model1.decision_function(roi)
                # print(">%s --> %0.4f" % (name, res[0]))
                results[name] = res[0]

        except Exception:
            print("something went wrong!")

        answer = max(results, key=results.get)
        scale = 0.5
        fontScale = min(endX - startX, endY - startY) / (25 / scale)
        cv2.putText(pil_image, answer, (startX, startY - int(fontScale)), cv2.FONT_HERSHEY_SIMPLEX, int(1),
                    (255, 255, 0), int(1), cv2.LINE_AA)

        if many:
            return answer
        else:
            return pil_image

    except Exception:
        print("something went wrong!")


def readHog(path_, models, many=None):
    results = {}
    boxes = {}
    models_dict = models
    pil_image = cv2.imread(path_)

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

            img = Image.open(path_).convert("L")
            image_array = np.array(img, "uint8")
            pic_ = image_array[startY:endY, startX:endX]
            pic = cv2.resize(pic_, (304, 304))

            dst = cv2.GaussianBlur(pic, (5, 5), cv2.BORDER_DEFAULT)
            fd, hog_image = hog(dst, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), visualize=True)
            roi = hog_image.flatten()

            roi = roi.reshape(1, -1)

            for name, model_ in models_dict.items():
                path_name = os.path.join(models_dir, model_)
                pickle_in = open(path_name, 'rb')
                model1 = pickle.load(pickle_in)
                pickle_in.close()
                res = model1.decision_function(roi)
                # print(">%s --> %0.4f" % (name, res[0]))
                results[name] = res[0]

        except Exception:
            print("something went wrong!")

        answer = max(results, key=results.get)
        scale = 0.5
        fontScale = min(endX - startX, endY - startY) / (25 / scale)
        cv2.putText(pil_image, answer, (startX, startY - int(fontScale)), cv2.FONT_HERSHEY_SIMPLEX, int(1),
                    (255, 255, 0), int(1), cv2.LINE_AA)

        if many:
            return answer
        else:
            return pil_image

    except Exception:
        print("something went wrong!")


def analysisSobel(models):
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            path = os.path.join(root, filename)
            answer = readSobel(path, models, True)
            print(">%s --> %s" % (filename, answer))


def analysisHog(models):
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            path = os.path.join(root, filename)
            answer = readHog(path, models, True)
            print(">%s --> %s" % (filename, answer))


def view(path):
    ocs_sobel = readSobel(path, OCS_sobel_models_dict)
    svc_sobel = readSobel(path, SVC_sobel_models_dict)
    svc_hog = readHog(path, SVC_hog_models_dict)
    ocs_hog = readHog(path, OCS_hog_models_dict)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].imshow(np.flipud(ocs_sobel[..., ::-1]), origin='lower')
    axis[0, 0].set_title('SOBEL OCS result')
    axis[0, 1].imshow(np.flipud(svc_sobel[..., ::-1]), origin='lower')
    axis[0, 1].set_title('SOBEL SVC result')
    axis[1, 0].imshow(np.flipud(ocs_hog[..., ::-1]), origin='lower')
    axis[1, 0].set_title('HOG OCS result')
    axis[1, 1].imshow(np.flipud(svc_hog[..., ::-1]), origin='lower')
    axis[1, 1].set_title('HOG SVC result')
    axis[0, 0].axes.get_xaxis().set_ticks([])
    axis[0, 0].axes.get_yaxis().set_ticks([])
    axis[0, 1].axes.get_xaxis().set_ticks([])
    axis[0, 1].axes.get_yaxis().set_ticks([])
    axis[1, 0].axes.get_xaxis().set_ticks([])
    axis[1, 0].axes.get_yaxis().set_ticks([])
    axis[1, 1].axes.get_xaxis().set_ticks([])
    axis[1, 1].axes.get_yaxis().set_ticks([])
    plt.show()


def analysis():
    print("OCS sobel")
    analysisSobel(OCS_sobel_models_dict)

    print("SVC sobel")
    analysisSobel(SVC_sobel_models_dict)

    print("OCS hog")
    analysisHog(OCS_hog_models_dict)

    print("SVC hog")
    analysisHog(SVC_hog_models_dict)
