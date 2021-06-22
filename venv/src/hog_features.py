import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2  # opencv
import pickle
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")

pics_data = os.path.join(pickles_dir, 'pics.pickle')
pickle_info = open(pics_data, 'rb')
key_data = pickle.load(pickle_info)
pickle_info.close()
data = []

for roi, label in key_data:
    dst = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)
    fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), visualize=True)
    roi = hog_image.flatten()
    data.append([roi, label])

data_path = os.path.join(pickles_dir, 'pic_hog_data_.pickle')

with open(data_path, 'wb') as f:
    pickle.dump(data, f)