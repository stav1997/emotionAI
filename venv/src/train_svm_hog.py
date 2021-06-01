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

dir_pickle_dict = {'angry':'mtcnn_angry_data.pickle', 'disgust':'mtcnn_disgust_data.pickle', 'happy':'mtcnn_happy_data.pickle', 'natural':'mtcnn_natural_data.pickle', 'sad':'mtcnn_sad_data.pickle', 'shock':'mtcnn_shock_data.pickle'}
# print(dir_pickle_dict.keys())
# print(dir_pickle_dict.values())
for key, value in dir_pickle_dict.items():
    data = []
    features = []
    labels = []
    filenames = []
    print("starting %s model training using data file: %s" % (key, value))
    # pickle_info = open(value, 'rb')
    # key_data = pickle.load(pickle_info)
    # pickle_info.close()
    # # print(key_data)
    # for roi, lable in key_data:
    #     fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    #     roi = hog_image.flatten()
    #     data.append([roi, lable])

    pickle_in = open(key+'_model_data_hog.pickle', 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()

    random.shuffle(data)
    for feature, label in data:
        features.append(feature)
        labels.append(1)
        # labels.append(label[0])
        filenames.append(label[1])
    # print(features)
    # print(labels)

    model_name = key+'_model'
    train_data, test_data, train_target, test_target = train_test_split(features, labels, train_size=0.8)
    # print(features)
    # print(train_data)
    # model_name = SVC(C=10, kernel='linear', degree=4, gamma=0.00001, decision_function_shape='ovo',
    #             class_weight='balanced', random_state=42)
    model_name = OneClassSVM(kernel='linear', gamma=0.005, nu=0.03)
    # model_name = OneClassSVM(kernel='linear', gamma=0.0005, nu=0.030389)

    # model_name.fit(features)
    model_name.fit(train_data)


    prediction = model_name.predict(train_data)
    # scores = model_name.score_samples(train_data)
    # # print(scores)
    # df = DataFrame(scores)
    # threshold = df.quantile(.03)
    # thresh = threshold.values[0]
    # print("threshold: ", thresh)


    print("accuracy train: ", metrics.accuracy_score(train_target, prediction))

    prediction1 = model_name.predict(test_data)
    print("accuracy test: ", metrics.accuracy_score(test_target, prediction1))

    # print(prediction)
    # anom_index = [train_data[i] for i, word in enumerate(prediction) if word==-1]
    # # print(anom_index)
    #
    # values = anom_index
    #
    # plt.scatter(train_data[:][0], train_data[:][1])
    # plt.scatter(values[:][0], values[:][1], color='r')
    # plt.show()

    # scores = model_name.score_samples(train_data)
    # # print(scores)
    # df = DataFrame(scores)
    # threshold = df.quantile(.03)
    # thresh = threshold.values[0]
    # print("threshold: ", thresh)
    # anom_index = [train_data[i] for i, score in enumerate(scores) if score<=thresh]
    # # print(anom_index)
    # plt.scatter(train_data[:][0], train_data[:][1])
    # plt.scatter(anom_index[:][0], anom_index[:][1], color='r')
    # plt.show()
    # scores = cross_val_score(model_name, features, labels, cv=10)
    # print(scores)
    # print("%s: %0.2f accuracy with a standard deviation of %0.2f" % (model_name, scores.mean(), scores.std()))

    # with open(key+'_model_data_hog.pickle', 'wb') as f:
    #     pickle.dump(data, f)

    with open(key+'_model.sav', 'wb') as f:
        pickle.dump(model_name, f)
