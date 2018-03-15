from sklearn import linear_model


def create():
    return linear_model.PassiveAggressiveClassifier(
            max_iter=1000, tol=1e-3)

