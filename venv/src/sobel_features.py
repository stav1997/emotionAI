import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
data = []

pics_data = os.path.join(pickles_dir, 'pics.pickle')
pickle_info = open(pics_data, 'rb')
key_data = pickle.load(pickle_info)
pickle_info.close()

for roi, label in key_data:
    dst = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)

    grad_x = cv2.Sobel(dst, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(dst, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_gard_x = cv2.convertScaleAbs(grad_x)
    abs_gard_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_gard_x, 0.5, abs_gard_y, 0.5, 0)

    roi = grad.flatten()
    data.append([roi, label])

data_path = os.path.join(pickles_dir, 'pic_sobel_data_.pickle')
with open(data_path, 'wb') as f:
    pickle.dump(data, f)
