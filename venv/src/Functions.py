import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2
from PIL import Image
import pickle
from skimage.feature import hog
import matplotlib.pyplot as plt
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
prototxt_path = os.path.join(BASE_DIR, 'model_data\\deploy.prototxt')
caffemodel_path = os.path.join(BASE_DIR, 'model_data\\weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

image_dir = os.path.join(BASE_DIR, "samples\\validation")
pickles_dir = os.path.join(BASE_DIR, "pickles")

scores_dict = {'hog_SVC': 'SVC_hog_scores.pickle', 'sobel_SVC': 'SVC_sobel_scores.pickle', 'hog_OCS': 'OCS_hog_scores.pickle', 'sobel_OCS': 'OCS_sobel_scores.pickle'}

OCS_sobel_models_dict = {'Angry': 'angry_OCS_sobel_model.sav', 'Disgust': 'disgust_OCS_sobel_model.sav',
                         'Happy': 'happy_OCS_sobel_model.sav', 'Natural': 'natural_OCS_sobel_model.sav',
                         'Sad': 'sad_OCS_sobel_model.sav', 'Shock': 'shock_OCS_sobel_model.sav'}

OCS_hog_models_dict = {'Angry': 'angry_OCS_hog_model.sav', 'Disgust': 'disgust_OCS_hog_model.sav',
                       'Happy': 'happy_OCS_hog_model.sav', 'Natural': 'natural_OCS_hog_model.sav',
                       'Sad': 'sad_OCS_hog_model.sav', 'Shock': 'shock_OCS_hog_model.sav'}

SVC_hog_models_dict = {'Angry': 'angry_SVC_hog_model.sav', 'Disgust': 'disgust_SVC_hog_model.sav',
                       'Happy': 'happy_SVC_hog_model.sav', 'Natural': 'natural_SVC_hog_model.sav',
                       'Sad': 'sad_SVC_hog_model.sav', 'Shock': 'Shock_SVC_hog_model.sav'}

SVC_sobel_models_dict = {'Angry': 'angry_SVC_sobel_model.sav', 'Disgust': 'disgust_SVC_sobel_model.sav',
                         'Happy': 'happy_SVC_sobel_model.sav', 'Natural': 'natural_SVC_sobel_model.sav',
                         'Sad': 'sad_SVC_sobel_model.sav', 'Shock': 'shock_SVC_sobel_model.sav'}


def boosting(path_, models, name=None, many=None):
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
            if name == 'hog':
                fd, hog_image = hog(dst, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), visualize=True)
                roi = hog_image.flatten()

            elif name == 'sobel':
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


def analysis():

    data = []
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            classification = {}
            path = os.path.join(root, filename)
            classification['filename'] = filename
            classification['OCS HOG'] = boosting(path, OCS_hog_models_dict, 'hog', True)
            classification['OCS Sobel'] = boosting(path, OCS_sobel_models_dict, 'sobel', True)
            classification['SVC HOG'] = boosting(path, SVC_hog_models_dict, 'hog', True)
            classification['SVC Sobel'] = boosting(path, SVC_sobel_models_dict, 'sobel', True)
            data.append(classification)
            print(classification)

def view(path):

    ocs_sobel = boosting(path, OCS_sobel_models_dict, 'sobel')
    svc_sobel = boosting(path, SVC_sobel_models_dict, 'sobel')
    svc_hog = boosting(path, SVC_hog_models_dict, 'hog')
    ocs_hog = boosting(path, OCS_hog_models_dict, 'hog')

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


def dataSplit(system_type, key, path):
    data_array = []
    features = []
    labels = []
    true_data = []
    false_data = []

    data_path = path
    pickle_in = open(data_path, 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    random.shuffle(data)

    for feature, label in data:
        if label == key:
            true_data.append([feature, 1])
        else:
            false_data.append([feature, -1])

    if system_type == 'ocs':
        data_ = true_data

    elif system_type == 'svc':
        n = len(true_data) - 80
        data_ = true_data + false_data[:n]

    for feature, label in data_:
        features.append(feature)
        labels.append(label)

    data_array.append(features)
    data_array.append(labels)

    return data_array


def showScoreResults():
    scores_hog_SVC = []
    names = ['Angry', 'Disgust', 'Happy', 'Natural', 'Sad', 'Shock']
    scores_sobel_SVC = []
    scores_sobel_OCS = []
    scores_hog_OCS = []
    scores = {}
    N = 6

    ind = np.arange(N)
    width = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for key in scores_dict:
        scores_ = []
        print(key)
        path = os.path.join(pickles_dir, scores_dict[key])
        pickle_info = open(path, 'rb')
        results = pickle.load(pickle_info)
        pickle_info.close()
        for name, score in results:
            scores_.append(np.mean(score)*100)
        scores[key] = scores_

    for key in scores:
        if key == 'hog_SVC':
            scores_hog_SVC = scores[key]
        if key == 'hog_OCS':
            scores_hog_OCS = scores[key]
        if key == 'sobel_SVC':
            scores_sobel_SVC = scores[key]
        if key == 'sobel_OCS':
            scores_sobel_OCS = scores[key]

    rects1 = ax.bar(ind, scores_hog_SVC, width, color='blue')
    rects2 = ax.bar(ind - width, scores_hog_OCS, width, color='red')
    rects3 = ax.bar(ind + width, scores_sobel_SVC, width, color='GREEN')
    rects4 = ax.bar(ind + width*2, scores_sobel_OCS, width, color='black')
    ax.set_ylabel('Accurecy in precentage')
    ax.set_xlabel('Emotions')

    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(names)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('HOG-SVC','HOG-OCS', 'SOBEL-SVC','SOBEL-OCS'), frameon=False, loc='upper left', ncol=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 1.9, 1.0* h, '%.1f' % float(h)+'%', ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.title('10-fold cross-validation results')
    plt.show()