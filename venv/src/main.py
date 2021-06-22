import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
from scores_and_results import showScoreResults
from OCS_hog import ocsHog
from OCS_sobel import ocsSobel
from SVC_sobel import svcSobel
from SVC_hog import svcHog

if __name__ == '__main__':

    path = "C:\\Users\\stav\\Desktop\\insta\\stav\\shock\\235_.jpg"

    res_sobel = ocsSobel(path)
    res_sobel1 = svcSobel(path)
    res_hog = svcHog(path)
    res_hog1 = ocsHog(path)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].imshow(np.flipud(res_sobel[...,::-1]), origin='lower')
    axis[0, 0].set_title('SOBEL OCS result')
    axis[0, 1].imshow(np.flipud(res_sobel1[...,::-1]), origin='lower')
    axis[0, 1].set_title('SOBEL SVC result')
    axis[1, 0].imshow(np.flipud(res_hog1[...,::-1]), origin='lower')
    axis[1, 0].set_title('HOG OCS result')
    axis[1, 1].imshow(np.flipud(res_hog[...,::-1]), origin='lower')
    axis[1, 1].set_title('HOG SVC result')
    plt.show()

    showScoreResults()
