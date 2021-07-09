import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from Functions import dataSplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
models_dir = os.path.join(BASE_DIR, "models")
data_path = os.path.join(pickles_dir, 'pic_sobel_data_.pickle')

models = []
results = []
data = []
features = []
labels = []
dir_dict = {'angry': 0, 'disgust': 1, 'happy': 2, 'natural': 3, 'sad': 4, 'shock': 5}

for key, value in dir_dict.items():
    data = dataSplit('svc', key, data_path)
    features = data[0]
    labels = data[1]

    model_name = SVC(C=10, kernel='linear', degree=4, gamma='scale', class_weight='balanced', probability=True)
    models.append([key, model_name, features, labels])

for name, model, features_, labels_ in models:
    print("starting %s model training" % name)

    scores = cross_val_score(model, features_, labels_, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
    results.append([name, scores])
    print('CV results: %s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    model.fit(features_, labels_)

    path_name = os.path.join(models_dir, name+'_SVC_sobel_model.sav')
    with open(path_name, 'wb') as f:
        pickle.dump(model, f)

path_ = os.path.join(pickles_dir, 'SVC_sobel_scores.pickle')
with open(path_, 'wb') as f:
    pickle.dump(results, f)