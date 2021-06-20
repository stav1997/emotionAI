import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
import pickle
import random
from sklearn.svm import SVC  # support vector classifier
from sklearn import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
names_hog = []
scores_hog = []
names_sobel = []
scores_sobel = []
results_hog = []
results_sobel = []

path = os.path.join(pickles_dir, 'SVC_hog_scores.pickle')
pickle_info = open(path, 'rb')
results_hog = pickle.load(pickle_info)
pickle_info.close()

path = os.path.join(pickles_dir, 'SVC_sobel_scores.pickle')
pickle_info = open(path, 'rb')
results_sobel = pickle.load(pickle_info)
pickle_info.close()

def showScoreResults():
    N = 6

    ind = np.arange(N)
    width = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2

    for name, score in results_hog:
        names_hog.append(name)
        scores_hog.append(np.mean(score))

    for name, score in results_sobel:
        names_sobel.append(name)
        scores_sobel.append(np.mean(score))

    rects1 = ax.bar(ind, scores_hog, width, color='blue')
    rects2 = ax.bar(ind + width, scores_sobel, width, color='red')

    ax.set_ylabel('Accurecy precentage')
    ax.set_xlabel('Emotions')

    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(names_sobel)
    ax.legend((rects1[0], rects2[0]), ('HOG', 'SOBEL'))

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.0* h, '%.4f' % float(h), ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    plt.title('10-fold cross-validation')
    plt.suptitle('SVC with HOG feature extraction', x = mid)
    plt.show()
