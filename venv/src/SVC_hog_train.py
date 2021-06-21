import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
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
from sklearn.ensemble import StackingClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
models_dir = os.path.join(BASE_DIR, "models")

models = []
results = []
names = []
roi_image_dir = os.path.join(BASE_DIR, "samples\\faces")
data = []

filenames = []
height = []
tick_label = []
dir_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}
models_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}
#
# pics_data = os.path.join(pickles_dir, 'pics.pickle')
# pickle_info = open(pics_data, 'rb')
# key_data = pickle.load(pickle_info)
# pickle_info.close()
#
# for roi, label in key_data:
#     dst = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)
#     fd, hog_image = hog(dst, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), visualize=True)
#     roi = hog_image.flatten()
#     data.append([roi, label])
#
data_path = os.path.join(pickles_dir, 'pic_hog_data_.pickle')
#
# with open(data_path, 'wb') as f:
#     pickle.dump(data, f)

pickle_in = open(data_path, 'rb')
data = pickle.load(pickle_in)
pickle_in.close()


for key, value in dir_dict.items():
    features = []
    labels = []
    true_data = []
    true_labels = []
    false_data = []
    false_labels = []

    random.shuffle(data)

    for feature, label in data:
        if label == key:
            true_data.append([feature, 1])
        else:
            false_data.append([feature, -1])

    print(len(true_data))
    data_ = true_data + false_data[:70]
    random.shuffle(data_)
    for feature, label in data_:
        features.append(feature)
        labels.append(label)

    model_name = SVC(C=10, kernel='linear', degree=4, gamma='scale', class_weight='balanced', probability=True)
    models.append([key, model_name, features, labels])

for name, model, features_, labels_ in models:
    print("starting %s model training" % name)

    scores = cross_val_score(model, features_, labels_, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
    results.append([name,scores])
    print('CV results: %s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    train_data, test_data, train_target, test_target = train_test_split(features_, labels_, train_size=0.2)
    model.fit(train_data, train_target)
    pred = model.predict(test_data)
    score = metrics.balanced_accuracy_score(test_target, pred)
    print('FIT results: %s %.3f' % (name, score))

    path_name = os.path.join(models_dir, name + '_SVC_hog_model.sav')
    with open(path_name, 'wb') as f:
        pickle.dump(model, f)

path_ = os.path.join(pickles_dir, 'SVC_hog_scores.pickle')
with open(path_, 'wb') as f:
    pickle.dump(results, f)
