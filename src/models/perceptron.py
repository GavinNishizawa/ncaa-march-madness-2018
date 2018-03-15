from sklearn import linear_model


def create():
    return linear_model.Perceptron(max_iter=1000, tol=1e-3)

