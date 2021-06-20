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
scores_hog_SVC = []
names_sobel = []
scores_sobel_SVC = []
scores_sobel_OCS = []
scores_hog_OCS = []

path = os.path.join(pickles_dir, 'SVC_hog_scores.pickle')
pickle_info = open(path, 'rb')
results_hog_SVC = pickle.load(pickle_info)
pickle_info.close()

path = os.path.join(pickles_dir, 'SVC_sobel_scores.pickle')
pickle_info = open(path, 'rb')
results_sobel_SVC = pickle.load(pickle_info)
pickle_info.close()

path = os.path.join(pickles_dir, 'OCS_hog_scores.pickle')
pickle_info = open(path, 'rb')
results_hog_OCS = pickle.load(pickle_info)
pickle_info.close()

path = os.path.join(pickles_dir, 'OCS_sobel_scores.pickle')
pickle_info = open(path, 'rb')
results_sobel_OCS = pickle.load(pickle_info)
pickle_info.close()

def showScoreResults():
    N = 6

    ind = np.arange(N)
    width = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2

    for name, score in results_hog_SVC:
        scores_hog_SVC.append(np.mean(score)*100)

    for name, score in results_sobel_SVC:
        names_sobel.append(name)
        scores_sobel_SVC.append(np.mean(score)*100)

    for name, score in results_hog_OCS:
        scores_hog_OCS.append(np.mean(score)*100)

    for name, score in results_sobel_OCS:
        scores_sobel_OCS.append(np.mean(score)*100)

    rects1 = ax.bar(ind, scores_hog_SVC, width, color='blue')
    rects2 = ax.bar(ind - width, scores_hog_OCS, width, color='red')
    rects3 = ax.bar(ind + width, scores_sobel_SVC, width, color='GREEN')
    rects4 = ax.bar(ind + width*2, scores_sobel_OCS, width, color='black')
    ax.set_ylabel('Accurecy in precentage')
    ax.set_xlabel('Emotions')

    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(names_sobel)
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('HOG-SVC','HOG-OCS', 'SOBEL-SVC','SOBEL-OCS'))

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 1.9, 1.0* h, '%.2f' % float(h)+'%', ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.title('10-fold cross-validation')
    plt.suptitle('SVC with HOG feature extraction', x = mid)
    plt.show()
