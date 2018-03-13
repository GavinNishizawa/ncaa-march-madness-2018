from sklearn import linear_model


def create_model():
    return linear_model.Perceptron(max_iter=1000, tol=1e-3)

def train(data):
    X = data[:, :2]
    Y = data[:, 2]
    model = create_model()
    model.fit(X, Y)
    return model


def test(test_data, model):
    return model.predict(test_data)


def main():
    print("Use src/run_model.py to run.")


if __name__ == "__main__":
    main()
