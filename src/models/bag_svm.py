from sklearn import svm, ensemble


def create():
    return ensemble.BaggingClassifier(
        svm.SVC(probability=True),
        max_samples=0.3,
        max_features=1.0)

