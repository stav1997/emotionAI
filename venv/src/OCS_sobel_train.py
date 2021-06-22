import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.model_selection import cross_val_score

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

data_path = os.path.join(pickles_dir, 'pic_sobel_data_.pickle')
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

    model_name = OneClassSVM(kernel='linear', gamma='scale', nu=0.1)
    # model_name = OneClassSVM(kernel='rbf', gamma=0.0005, nu=0.05)

    false_train_data, false_test_data, false_train_target, false_test_target = train_test_split(false_data, false_labels, train_size=0.09)
    # false_train_data, false_test_data, false_train_target, false_test_target = train_test_split(false_data, false_labels, train_size=0.005)


    train_data = true_data + false_train_data
    train_target = true_labels + false_train_target

    models.append([key, model_name, true_data, true_labels])
    # models.append([key, model_name, train_data, train_target])

for name, model, train_x, train_y in models:
    scores = cross_val_score(model, train_x, train_y, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
    results.append([name, scores])

    print('CV results: %s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    model.fit(train_x, train_y)
    pred = model.predict(train_x)
    score = metrics.balanced_accuracy_score(train_y, pred)
    print('FIT results: %s %.3f' % (name, score))

    # path_name = os.path.join(models_dir, name + '_OCS_sobel_model.sav')
    # with open(path_name, 'wb') as f:
    #     pickle.dump(model, f)

path_ = os.path.join(pickles_dir, 'OCS_sobel_scores.pickle')
with open(path_, 'wb') as f:
    pickle.dump(results, f)

