from sklearn import ensemble, naive_bayes


def create():
    return ensemble.BaggingClassifier(
        naive_bayes.GaussianNB(),
        max_samples=0.3,
        max_features=1.0)

