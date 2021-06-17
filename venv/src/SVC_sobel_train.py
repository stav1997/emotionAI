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
from sklearn.calibration import CalibratedClassifierCV

from sklearn.svm import OneClassSVM
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from pca import pca
from sklearn.feature_selection import SelectFromModel
from skimage.feature import canny
from sklearn.ensemble import StackingClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
models_dir = os.path.join(BASE_DIR, "models")

models = []
results = []
names = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
roi_image_dir = os.path.join(BASE_DIR, "samples\\faces")
data = []
features = []
labels = []
filenames = []
dir_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}
models_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}

# pics_data = os.path.join(pickles_dir, 'pics.pickle')
#
# pickle_info = open(pics_data, 'rb')
# key_data = pickle.load(pickle_info)
# pickle_info.close()
#
# for roi, label in key_data:
#     dst = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)
#
#     grad_x = cv2.Sobel(dst, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#     grad_y = cv2.Sobel(dst, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#     abs_gard_x = cv2.convertScaleAbs(grad_x)
#     abs_gard_y = cv2.convertScaleAbs(grad_y)
#     grad = cv2.addWeighted(abs_gard_x, 0.5, abs_gard_y, 0.5, 0)
#
#     roi = grad.flatten()
#     data.append([roi, label])
data_path = os.path.join(pickles_dir, 'pic_sobel_data_.pickle')

# with open(data_path, 'wb') as f:
#     pickle.dump(data, f)

# pickle_in = open('pic_sobel_data_.pickle', 'rb')
pickle_in = open(data_path, 'rb')
data = pickle.load(pickle_in)
pickle_in.close()


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

    model_name = SVC(C=10, kernel='linear', degree=4, gamma=0.00001, class_weight='balanced', probability=True)

    false_train_data, false_test_data, false_train_target, false_test_target = train_test_split(false_data, false_labels, train_size=0.17)
    true_train_data, true_test_data, true_train_target, true_test_target = train_test_split(true_data, true_labels, train_size=0.9)

    train_data = true_train_data + false_train_data
    train_target = true_train_target + false_train_target

    test_data = true_test_data + false_test_data
    test_target = true_test_target + false_test_target

    models.append([key, model_name, train_data, train_target, test_data[:30], test_target[:30]])

for name, model, train_x, train_y, test_x, test_y in models:
    # scores = cross_val_score(model, train_x, train_y, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
    # results.append(scores)
    # names.append(name)
    # print('CV results: %s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    model.fit(train_x, train_y)
    pred = model.predict(train_x)
    score = metrics.balanced_accuracy_score(train_y, pred)
    print('FIT results: %s %.3f' % (name, score))
    path_name = os.path.join(models_dir, name+'_SVC_sobel_model.sav')
    with open(path_name, 'wb') as f:
        pickle.dump(model, f)

    # clf = CalibratedClassifierCV(base_estimator=model, cv=5)
    # clf.fit(train_x, train_y)
    # pred = clf.predict(train_x)
    # score = metrics.balanced_accuracy_score(train_y, pred)
    # print('FIT results: %s %.3f' % (name, score))

    # res = model.decision_function(train_x)
    #
    # print(res)
    # print(train_x[:][0])
    # print(train_x[:][1])
    #
    # print(train_y)
    # plt.scatter(res, train_y, c=train_y, s=30, cmap=plt.cm.Paired)
    #
    # # plot the decision function
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    #
    # # create grid to evaluate model
    # xx = np.linspace(xlim[0], xlim[1], 30)
    # yy = np.linspace(ylim[0], ylim[1], 30)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = model.decision_function(xy).reshape(XX.shape)
    #
    # # plot decision boundary and margins
    # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    #
    # # plot support vectors
    # ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    #
    # plt.show()