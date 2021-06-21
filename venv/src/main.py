import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np

from scores_and_results import showScoreResults
from OCS_hog import ocsHog
from OCS_sobel import ocsSobel
from matplotlib.gridspec import GridSpec

from SVC_sobel import svcSobel
from SVC_hog import svcHog

if __name__ == '__main__':

    # path = "C:\\Users\\stav\\Desktop\\insta\\shock\\b69dbdcdaab3018db034125ea72d942e (1).jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\happy\\132_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\angry\\2_.jpg"
    #
    # # path = "C:\\Users\\stav\\Desktop\\final year project\\faces\\s.jpg"
    #
    # # path = "C:\\Users\\stav\\PycharmProjects\\finalProject\\venv\\src\\samples\\validation\\shock\\download (5).jpg"
    # res_sobel = ocsSobel(path)
    # res_sobel1 = svcSobel(path)
    # res_hog = svcHog(path)
    # res_hog1 = ocsHog(path)
    #
    # figure, axis = plt.subplots(2, 2)
    #
    # axis[0, 0].imshow(np.flipud(res_sobel[...,::-1]), origin='lower')
    # axis[0, 0].set_title('SOBEL OCS result')
    # axis[0, 1].imshow(np.flipud(res_sobel1[...,::-1]), origin='lower')
    # axis[0, 1].set_title('SOBEL SVC result')
    # axis[1, 0].imshow(np.flipud(res_hog1[...,::-1]), origin='lower')
    # axis[1, 0].set_title('HOG OCS result')
    # axis[1, 1].imshow(np.flipud(res_hog[...,::-1]), origin='lower')
    # axis[1, 1].set_title('HOG SVC result')
    # plt.show()

    showScoreResults()
    # print(res)
    # plt.imshow(np.flipud(res), origin='lower')
    #
    # plt.show()
