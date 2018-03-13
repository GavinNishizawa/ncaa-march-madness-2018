from sklearn import svm


def train(data):
    X = data[:, :2]
    Y = data[:, 2]
    model = svm.SVC()
    model.fit(X, Y)
    return model


def test(test_data, model):
    return model.predict(test_data)


def main():
    print("Use src/run_model.py to run.")


if __name__ == "__main__":
    main()
