import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from scores_and_results import showScoreResults
from Functions import view, analysis

if __name__ == '__main__':

    path = "C:\\Users\\stav\\Desktop\\insta\\stav\\pics\\angry\\shock_9.jpg"
    view(path)
    analysis()
    showScoreResults()
