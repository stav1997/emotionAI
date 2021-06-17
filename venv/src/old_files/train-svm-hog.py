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
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import cross_val_score



data = []
pictures = []
features = []
labels = []
filenames = []
current_id = 0
label_id = {}
categories = []
start = time.time()
pickle_in = open('mtcnn_data.pickle', 'rb')
pictures = pickle.load(pickle_in)
pickle_in.close()
for roi, lable in pictures:
    fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    roi = hog_image.flatten()
    data.append([roi, lable])

# pickle_in = open('data_hog.pickle', 'rb')
# data = pickle.load(pickle_in)
# pickle_in.close()

# random.shuffle(data)
for feature, label in data:
    features.append(feature)
    labels.append(label[0])
    filenames.append(label[1])


# train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25)

model = SVC(C=10, kernel='linear', degree=4, gamma=0.00001, decision_function_shape='ovo', class_weight='balanced', random_state=42)
scores = cross_val_score(model, features, labels, cv=10)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# model.fit(train_x, train_y)
# pred = model.predict(test_x)
# print(metrics.classification_report(test_y, pred))
# accuracy = metrics.balanced_accuracy_score(test_y, pred)
# # print("accuracy: ", accuracy)
# print("prediction is:", pred[0])
# emo = test_x[0].reshape(480, 480)
# # print("filename is:", test_x[0])
#
# plt.imshow(emo, cmap='gray')
# plt.show()

with open("data_hog.pickle", 'wb') as f:
    pickle.dump(data, f)

# with open("svm_hog.pickle", 'wb') as f:
#     pickle.dump(labels, f)

with open("model_svm_hog.sav", 'wb') as f:
    pickle.dump(model, f)

stop = time.time()
print(f"Training time: {stop - start}s")

