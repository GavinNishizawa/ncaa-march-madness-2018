import os
import numpy as np
from read_data import load_data
from save_data import save_object, load_object
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from seed_pred import get_seed_data
from split_data import split_data
from save_data import save_object, load_object
from model_man import get_models, train, test, predict
models = get_models()
from models import voting_hard, voting_soft, voting_weighted


def plot(X, Y, c_fn):
    # define colors array
    colors = np.array([c_fn(v) for v in Y])

    plt.scatter(X[:,0], X[:,1], c=colors, s=0.5)
    plt.tight_layout()


def plot_model(train, test, model_test_output):
    # plot training data in blue and orange
    c_fn = lambda v: "blue" if v == 1 else "orange"
    plot(train[:, :2], train[:, 2], c_fn)
    plt.show()
    plt.close()

    # plot test data in green and red
    c_fn = lambda v: "green" if v == 1 else "red"
    plot(test[:, :2], model_test_output, c_fn)

    plt.show()
    plt.close()


def run_prediction(m_name, data):
    train_data = data["train_data"]
    train_target = data["train_target"]
    test_data = data["test_data"]

    if m_name == "voting_hard":
        model = voting_hard
    elif m_name == "voting_soft":
        model = voting_soft
    elif m_name == "voting_weighted":
        model = voting_weighted
    else:
        model = models[m_name]


    trained_model = train(model.create(), train_data, train_target)

    ps = predict(trained_model, test_data)
    return ps


def run_model(m_name, data, retrain=False, verbose=False):
    train_data = data["train_data"]
    train_target = data["train_target"]
    test_data = data["test_data"]
    test_target = data["test_target"]

    if m_name == "voting_hard":
        model = voting_hard
    elif m_name == "voting_soft":
        model = voting_soft
    elif m_name == "voting_weighted":
        model = voting_weighted
    else:
        model = models[m_name]

    # trained_model filename
    tm_fn = os.path.join("src","models","trained_"+m_name+"_model")

    # load trained model if it exists
    trained_model = load_object(tm_fn)

    if retrain or trained_model == None:
        # train model on training data
        trained_model = train(model.create(), train_data, train_target)
        save_object(tm_fn, trained_model)

    test_results = test(trained_model, test_data)

    # calc accuracy
    test_accuracy = metrics.accuracy_score(test_target, test_results)
    test_log_loss = metrics.log_loss(test_target, test_results)
    test_f1 = metrics.f1_score(test_target, test_results)

    # record accuracy
    with open(tm_fn+"_results.csv",'a') as res_f:
        res_f.write("\n%f,%f,%f" % (test_accuracy, test_log_loss, test_f1))

    if verbose:
        #test_precision = metrics.precision_score(test_target, test_results)
        #test_recall = metrics.recall_score(test_target, test_results)
        test_report = metrics.classification_report(test_target, test_results)

        print("\nResults for",m_name,":")
        print("\tLog loss : %f" % test_log_loss)
        print("\tAccuracy : %0.5f" % test_accuracy)
        #print("\tRecall   : %0.5f" % test_recall)
        #print("\tPrecision: %0.5f" % test_precision)
        print("\tF1 score : %0.5f" % test_f1)
        print("\nReport:\n", test_report,"\n")

    else:
        print("Accuracy for",m_name,": %0.5f" % test_accuracy)


    return train_data, test_data, test_results


def apply_pca(t_data):
    # apply PCA and whiten
    pca = PCA(whiten=True)
    t_data["train_data"] = pca.fit_transform(t_data["train_data"])
    t_data["test_data"] = pca.transform(t_data["test_data"])

    return t_data


def main(retrain=False, verbose=False):

    # t_data filename
    td_fn = os.path.join("data","train_test_split")

    t_data = load_object(td_fn)

    if retrain or t_data == None:
        # create training/testing dataset
        data = load_data()
        seed_data = get_seed_data(data)

        # split data into training 70% and testing 30%
        train_data, test_data = split_data(seed_data.values, 0.7)

        # split target from data
        t_data = {
            "train_target": train_data[:, -1],
            "train_data": train_data[:, :-1],
            "test_target": test_data[:, -1],
            "test_data": test_data[:, :-1]
            }

        # save to file
        save_object(td_fn, t_data)

        retrain=True

    t_data = apply_pca(t_data)

    # run training and testing of each of the models on the dataset
    for key in models.keys():
        run_model(key, t_data, retrain, verbose)

    #train, test, test_predict = run_model("knn", t_data)
    #plot_model(train, test, test_predict)

    run_model("voting_hard", t_data, retrain, verbose)
    run_model("voting_soft", t_data, retrain, verbose)
    run_model("voting_weighted", t_data, retrain, verbose)


if __name__ == "__main__":
    main(True, True)

