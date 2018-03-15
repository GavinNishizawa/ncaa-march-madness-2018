from sklearn import svm


def create():
    return svm.SVC(probability=True)

