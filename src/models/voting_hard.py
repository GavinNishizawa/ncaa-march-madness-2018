from sklearn import ensemble
from model_man import get_models


def create():
    models = get_models()
    v_models = [(key, models[key].create()) for key in models.keys()]

    return ensemble.VotingClassifier(
            estimators=v_models,
            voting="hard"
            )

