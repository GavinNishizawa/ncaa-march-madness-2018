from sklearn import ensemble
from get_models import get_models


def create_model():
    models = get_models()
    del models["huber"]
    del models["perceptron"]
    del models["pass_aggr"]

    v_models = [(key, models[key].create_model()) for key in models.keys()]

    return ensemble.VotingClassifier(
            estimators=v_models,
            voting="soft"
            )

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
