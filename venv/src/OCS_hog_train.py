import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2  # opencv
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
models_dir = os.path.join(BASE_DIR, "models")

models = []
results = []
names = []
roi_image_dir = os.path.join(BASE_DIR, "samples\\faces")
data = []
features = []
labels = []
filenames = []
dir_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}
models_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}

data_path = os.path.join(pickles_dir, 'pic_hog_data_.pickle')
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

    # data_ = true_data + false_data[:10]
    data_ = true_data

    random.shuffle(data_)
    for feature, label in data_:
        features.append(feature)
        labels.append(label)

    # model_name = OneClassSVM(kernel='linear', gamma='scale', nu=0.13)
    model_name = OneClassSVM(kernel='linear', gamma=0.0005, nu=0.1)

    models.append([key, model_name, features, labels])

for name, model, features_, labels_ in models:
    print("starting %s model training" % name)

    scores = cross_val_score(model, features_, labels_, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
    results.append([name, scores])
    print('CV results: %s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    model.fit(features_, labels_)

    path_name = os.path.join(models_dir, name + '_OCS_hog_model.sav')
    with open(path_name, 'wb') as f:
        pickle.dump(model, f)

path_ = os.path.join(pickles_dir, 'OCS_hog_scores.pickle')
with open(path_, 'wb') as f:
    pickle.dump(results, f)
