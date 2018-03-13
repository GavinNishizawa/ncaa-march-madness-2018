import random as r
import numpy as np
from read_data import load_data
from save_data import save_object, load_object
from sklearn import metrics
import matplotlib.pyplot as plt
from seed_pred import get_seed_data
import knn
import svm
import log_reg
import perceptron
import pass_aggr
import huber


models = {
    "knn": knn, "svm": svm,
    "log_reg": log_reg,
    "huber": huber,
    "perceptron": perceptron,
    "pass_aggr": pass_aggr
}


def plot(X, Y, c_fn):
    # define colors array
    colors = np.array([c_fn(v) for v in Y])

    plt.scatter(X[:,0], X[:,1], c=colors, s=0.5)
    plt.tight_layout()


def split_data(data, ratio):

    # split data into wins and losses assume wins then losses
    ld = int(len(data)/2)
    wins = data[:ld]
    losses = data[ld:]

    # shuffle data
    #r.seed(9001)
    for i in range(r.randint(3,10)):
        r.shuffle(wins)
        r.shuffle(losses)

    # split each set into sections according to ratio
    s_ind_w = int(len(wins)*ratio)
    s_ind_l = int(len(losses)*ratio)

    # create a training and testing set with wins and losses
    train = np.vstack((wins[:s_ind_w], losses[:s_ind_l]))
    test = np.vstack((wins[s_ind_w:], losses[s_ind_l:]))

    # shuffle the wins and losses in each set
    r.shuffle(train)
    r.shuffle(test)

    return train, test


def plot_model(train, test, model_test_output):
    # plot training data in blue and orange
    c_fn = lambda v: "blue" if v == 1 else "orange"
    plot(train[:, :2], train[:, 2], c_fn)

    # plot test data in green and red
    c_fn = lambda v: "green" if v == 1 else "red"
    plot(test[:, :2], model_test_output, c_fn)

    plt.show()


def _run_model(m_name, train_data, test_data):
    model = models[m_name]

    # train model on training data
    trained_model = model.train(train_data)
    test_results = model.test(test_data[:,:2], trained_model)

    # calc accuracy
    test_accuracy = metrics.accuracy_score(test_data[:,2], test_results)
    print("Accuracy for",m_name,":", test_accuracy)

    return train_data, test_data, test_results


def run_model(m_name):
    data = load_data()
    seed_data = get_seed_data(data)
    # split data into training 70% and testing 30%
    train_data, test_data = split_data(seed_data.values, 0.7)

    return _run_model(m_name, train_data, test_data)


def main():
    for key in models.keys():
        run_model(key)
    #plot_model(train, test, test_predict)


if __name__ == "__main__":
    main()

