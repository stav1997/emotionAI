import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
# from lbp_mtcnn import lbpMtcnn
# from lbp_haar import lbpHaar
# from svm_mtcnn import svmMtcnn
from svm_hog import svmHog


if __name__ == '__main__':
    path = "C:\\Users\\stav\\Desktop\\final\\venv\\src\\samples\\train\\sad\\215_.jpg"

    # path = "C:\\Users\\stav\\Desktop\\final year project\\faces\\s.jpg"

    # path = "C:\\Users\\stav\\PycharmProjects\\finalProject\\venv\\src\\samples\\validation\\shock\\download (5).jpg"
    # lbp = lbpMtcnn(path)
    # lbp = lbpHaar(path)
    # lbp = svmMtcnn(path)
    lbp = svmHog(path)

    plt.imshow(np.flipud(lbp), origin='lower')

    plt.show()
