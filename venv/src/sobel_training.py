import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
import time
import random
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # support vector classifier
from sklearn.svm import OneClassSVM
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from pca import pca
from sklearn.feature_selection import SelectFromModel
from skimage.feature import canny

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
roi_image_dir = os.path.join(BASE_DIR, "samples\\faces")
data = []
features = []
labels = []
filenames = []
path1 = "C:\\Users\\stav\\Desktop\\test.jpg"
dir_dict = {'angry':0, 'disgust':1, 'happy':2, 'natural':3, 'sad':4, 'shock':5}

pickle_info = open('pics.pickle', 'rb')
key_data = pickle.load(pickle_info)
pickle_info.close()

for roi, label in key_data:
    # print("roi: ", roi)
    dst = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)
    # plt.imshow(np.hstack((roi, dst)))
    # plt.show()
    # print("dst: ", dst)
    grad_x = cv2.Sobel(dst, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(dst, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_gard_x = cv2.convertScaleAbs(grad_x)
    abs_gard_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_gard_x, 0.5, abs_gard_y, 0.5, 0)

    # print("grad: ", grad)

    # edges = canny(
    #     image=dst,
    #     sigma=1,
    #     low_threshold=0.0,
    #     high_threshold=1.0,
    # )

    # plt.imshow(grad)
    # plt.show()
    roi = grad.flatten()
    # print("grad flatten: ", roi)

    data.append([roi, label])

with open('model_sobel.pickle', 'wb') as f:
    pickle.dump(data, f)

# pickle_in = open('model_sobel.pickle', 'rb')
# data = pickle.load(pickle_in)
# pickle_in.close()

for key, value in dir_dict.items():

    true_data = []
    true_labels = []
    false_data = []
    false_labels = []

    random.shuffle(data)
    for feature, label in data:
        if label == key:
            true_data.append(feature)
            true_labels.append(1)
        else:
            false_data.append(feature)
            false_labels.append(-1)


    print("starting %s model training using data file: %s" % (key, value))

    model_name = key+'_model'
    model_name = OneClassSVM(kernel='linear', gamma=0.0005, nu=0.05)
    false_train_data, false_test_data, false_train_target, false_test_target = train_test_split(false_data, false_labels, train_size=0.05)
    true_train_data, true_test_data, true_train_target, true_test_target = train_test_split(true_data, true_labels, train_size=0.9)


    train_data = true_train_data + false_train_data
    train_target = true_train_target + false_train_target

    test_data = true_test_data + false_test_data
    test_target = true_test_target + false_test_target

    model_name.fit(train_data)

    prediction = model_name.predict(train_data)
    print("accuracy train: ", metrics.accuracy_score(train_target, prediction))

    prediction1 = model_name.predict(test_data[:30])
    print("accuracy test: ", metrics.accuracy_score(test_target[:30], prediction1))


    anom_index = [train_data[i] for i, word in enumerate(prediction) if word==-1]
    values = anom_index
    plt.suptitle(key)
    plt.figure(1)
    plt.subplot(121)
    plt.title("without threshold")


    plt.scatter(train_data[:][0], train_data[:][1])
    plt.scatter(values[:][0], values[:][1], color='r')

    scores = model_name.score_samples(train_data)
    df = DataFrame(scores)
    threshold = df.quantile(.03)
    thresh = threshold.values[0]
    anom_index = [train_data[i] for i, score in enumerate(scores) if score<=thresh]
    plt.subplot(122)
    plt.title("with threshold")


    plt.scatter(train_data[:][0], train_data[:][1])
    plt.scatter(anom_index[:][0], anom_index[:][1], color='r')
    # plt.show()

    scores = cross_val_score(model_name, train_data, train_target, cv=10, scoring="accuracy")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("\n")

    # # with open(key+'_model_sobel.pickle', 'wb') as f:
    # #     pickle.dump(data, f)
    #
    # with open(key+'_model.sav', 'wb') as f:
    #     pickle.dump(model_name, f)
