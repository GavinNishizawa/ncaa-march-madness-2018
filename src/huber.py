from sklearn import linear_model
import numpy as np


def train(data):
    X = data[:, :2]
    Y = data[:, 2]
    model = linear_model.HuberRegressor()
    model.fit(X, Y)
    return model


def test(test_data, model):
    return np.array(list(map((lambda x: round(x)), model.predict(test_data))))


def main():
    print("Use src/run_model.py to run.")


if __name__ == "__main__":
    main()
