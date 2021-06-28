import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
from scores_and_results import showScoreResults
# from OCS_hog import ocsHog
# from OCS_sobel import ocsSobel
# from SVC_sobel import svcSobel
# from SVC_hog import svcHog
from ocs_hog_ import ocsHog
from ocs_sobel_ import ocsSobel
from svc_hog_ import svcHog
from svc_sobel_ import svcSobel

if __name__ == '__main__':

    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\__15_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\__59_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\_166_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\20_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\115_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\180_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\230_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\280_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\298_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\463_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\496_.jpg"
    # path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\c_74_.jpg"


    # ocs_sobel = ocsSobel(path)
    # svc_sobel = svcSobel(path)
    # svc_hog = svcHog(path)
    # ocs_hog = ocsHog(path)
    # ocs_hog = ocsHog()
    # ocs_sobel = ocsSobel()
    # svc_hog = svcHog()
    # svc_sobel = svcSobel()

    # figure, axis = plt.subplots(2, 2)
    #
    # axis[0, 0].imshow(np.flipud(ocs_sobel[..., ::-1]), origin='lower')
    # axis[0, 0].set_title('SOBEL OCS result')
    # axis[0, 1].imshow(np.flipud(svc_sobel[..., ::-1]), origin='lower')
    # axis[0, 1].set_title('SOBEL SVC result')
    # axis[1, 0].imshow(np.flipud(ocs_hog[..., ::-1]), origin='lower')
    # axis[1, 0].set_title('HOG OCS result')
    # axis[1, 1].imshow(np.flipud(svc_hog[..., ::-1]), origin='lower')
    # axis[1, 1].set_title('HOG SVC result')
    # plt.show()

    showScoreResults()
