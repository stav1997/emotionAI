import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from Functions import view, analysis, showScoreResults

if __name__ == '__main__':

    path = "C:\\Users\\stav\\Desktop\\yesterday\\venv\\src\\samples\\validation\\shock_9.jpg"
    view(path)
    analysis()
    showScoreResults()

