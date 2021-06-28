import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2  # opencv
import pickle
import time
from skimage.feature import hog
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

start = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(BASE_DIR, "pickles")
data = []
pics_data = os.path.join(pickles_dir, 'pics.pickle')
pickle_info = open(pics_data, 'rb')
key_data = pickle.load(pickle_info)
pickle_info.close()

for roi, label in key_data:
    dst = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)
    fd, hog_image = hog(dst, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(4, 4), visualize=True)
    plt.imshow()
    roi = hog_image.flatten()
    data.append([roi, label])

data_path = os.path.join(pickles_dir, 'pic_hog_data_.pickle')

with open(data_path, 'wb') as f:
    pickle.dump(data, f)

stop = time.time()
print(f"HOG time: {stop - start}s")

#
#
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import cv2  # opencv
# import pickle
# import time
# from skimage.feature import hog
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
#
# start = time.time()
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# pickles_dir = os.path.join(BASE_DIR, "pickles")
# data = []
# roi_image_dir = "C:\\Users\\stav\\Desktop"
#
# path = "C:\\Users\\stav\\Desktop\\yesterday\\venv\\src\\samples\\faces\\angry\\136_.jpg"
# img = Image.open(path).convert("L")
# image_array = np.array(img, "uint8")
# cv2.imwrite(roi_image_dir + '\\gray_136_.jpg', image_array)
#
# dst = cv2.GaussianBlur(image_array, (5, 5), cv2.BORDER_DEFAULT)
# cv2.imwrite(roi_image_dir + '\\gray_blur_136_.jpg', dst)
#
# grad_x = cv2.Sobel(dst, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
# grad_y = cv2.Sobel(dst, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
# abs_gard_x = cv2.convertScaleAbs(grad_x)
# cv2.imwrite(roi_image_dir + '\\gray_blur_sobel_x_136_.jpg', abs_gard_x)
#
# abs_gard_y = cv2.convertScaleAbs(grad_y)
# cv2.imwrite(roi_image_dir + '\\gray_blur_sobel_y_136_.jpg', abs_gard_y)
#
# grad = cv2.addWeighted(abs_gard_x, 0.5, abs_gard_y, 0.5, 0)
# cv2.imwrite(roi_image_dir + '\\gray_blur_sobel_136_.jpg', grad)