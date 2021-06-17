import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
# from src.svm_hog import svmHog
# from lbp_mtcnn import lbpMtcnn
# from lbp_haar import lbpHaar
# from svm_mtcnn import svmMtcnn
from SVC_sobel import svcSobel

if __name__ == '__main__':
    path = "C:\\Users\\stav\\Desktop\\insta\\happy\\73.jpg"

    # path = "C:\\Users\\stav\\Desktop\\final year project\\faces\\s.jpg"

    # path = "C:\\Users\\stav\\PycharmProjects\\finalProject\\venv\\src\\samples\\validation\\shock\\download (5).jpg"
    # lbp = lbpMtcnn(path)
    # lbp = lbpHaar(path)
    # lbp = svmMtcnn(path)
    # lbp = svmHog(path)
    res = svcSobel(path)
    print(res)
    # plt.imshow(np.flipud(res), origin='lower')
    #
    # plt.show()
