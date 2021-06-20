import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
# from src.svm_hog import svmHog
# from lbp_mtcnn import lbpMtcnn
# from lbp_haar import lbpHaar
# from svm_mtcnn import svmMtcnn
from scores_and_results import showScoreResults
from OCS_sobel import ocsHog

from SVC_sobel import svcSobel
from SVC_hog import svcHog

if __name__ == '__main__':

    # path = "C:\\Users\\stav\\Desktop\\insta\\shock\\b69dbdcdaab3018db034125ea72d942e (1).jpg"
    path = "C:\\Users\\stav\\Desktop\\insta\\stav\\happy\\132_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\shock\\9b03bc5f-8cb9-4889-bea7-5c3609c9b323.jpg"

    # path = "C:\\Users\\stav\\Desktop\\final year project\\faces\\s.jpg"

    # path = "C:\\Users\\stav\\PycharmProjects\\finalProject\\venv\\src\\samples\\validation\\shock\\download (5).jpg"

    #
    res_hog = svcHog(path)
    plt.suptitle("RESULTS:")
    plt.figure(1)
    plt.subplot(121)
    plt.title("HOG SVC result")
    plt.imshow(np.flipud(res_hog[...,::-1]), origin='lower')

    res_sobel = ocsHog(path)
    plt.subplot(122)
    plt.title("HOG OCS result")
    plt.imshow(np.flipud(res_sobel[...,::-1]), origin='lower')
    plt.show()
    # showScoreResults()
    # print(res)
    # plt.imshow(np.flipud(res), origin='lower')
    #
    # plt.show()
