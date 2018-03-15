from sklearn import ensemble
from model_man import get_models
from read_results import load_data, get_averages


def create():
    models = get_models()
    del models["perceptron"]
    del models["pass_aggr"]

    data = load_data()
    data = get_averages(data)
    avgs = {}

    for key in data.keys():
        # remove trained_ (8) and _model_results (14)
        nk = key[8:-14]
        avgs[nk] = data[key]

    key_order = models.keys()

    v_models = [(key, models[key].create()) for key in key_order]

    # (accuracy*precision)/(log loss**2)
    v_weights = [(avgs[key][0]*avgs[key][2])/(avgs[key][1]**2)
        for key in key_order]

    return ensemble.VotingClassifier(
            estimators=v_models,
            voting="soft",
            weights=v_weights
            )

