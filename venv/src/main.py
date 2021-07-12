import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from Functions import view, analysis, showScoreResults
# import face_extraction
# import sobel_features
# import hog_features
# import OCS_hog_train
# import OCS_sobel_train
# import SVC_hog_train
# import SVC_sobel_train

if __name__ == '__main__':

    path = "C:\\Users\\stav\\Desktop\\yesterday\\venv\\src\\samples\\validation\\shock_9.jpg"
    view(path)
    analysis()
    showScoreResults()

